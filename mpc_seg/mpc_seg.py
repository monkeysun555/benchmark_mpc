import os
import logging
import numpy as np
import live_player_seg as live_player
import live_server_seg as live_server
import load
import mpc_solver_seg as mpc

IF_NEW = 1
IF_ALL_TESTING = 1		# IF THIS IS 1, IF_NEW MUST BE 1
# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]

RANDOM_SEED = 13
RAND_RANGE = 1000
MS_IN_S = 1000.0
KB_IN_MB = 1000.0	# in ms
INIT_BW = BITRATE[0]

SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
# CHUNK_DURATION = 200.0
# CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
# CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION

# Initial buffer length on server side
SERVER_START_UP_TH = 2000.0											# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0											# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL			# Accumulate latency upperbound

ACTION_REWARD = 1.0 	
REBUF_PENALTY = 3.0				# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 5.0 
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 3.0			# not included
# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1			# For 1
# NORMAL_PLAYING = 1.0			# For 0
# SLOW_PLAYING = 0.9			# For -1

TEST_DURATION = 100				# Number of testing <===================== Change length here
RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others
MPC_STEP = 5
# bitrate number is 6, no bin

if not IF_NEW:
	DATA_DIR = '../../bw_traces/'
	TRACE_NAME = '70ms_loss0.5_m5.txt'
else:
	DATA_DIR = '../../new_traces/test_sim_traces/'
	TRACE_NAME = 'norway_car_10'

if not IF_ALL_TESTING:
	LOG_FILE_DIR = './test_results'
	LOG_FILE = LOG_FILE_DIR + '/MPCSEG_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'
else:
	LOG_FILE_DIR = './all_test_results'
	LOG_FILE = LOG_FILE_DIR + '/MPCSEG_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'

def ReLU(x):
	return x * (x > 0)

def record_tp(tp_trace, time_trace, starting_time_idx, duration):
	tp_record = []
	time_record = []
	offset = 0
	time_offset = 0.0
	num_record = int(np.ceil(duration/SEG_DURATION))
	for i in range(num_record):
		if starting_time_idx + i + offset >= len(tp_trace):
			offset = -len(tp_trace)
			time_offset += time_trace[-1]
		tp_record.append(tp_trace[starting_time_idx + i + offset])
		time_record.append(time_trace[starting_time_idx + i + offset] + time_offset)
	return tp_record, time_record

def new_record_tp(tp_trace, time_trace, starting_time_idx, duration):
	# print starting_time_idx
	# print duration
	start_time = time_trace[starting_time_idx]
	tp_record = []
	time_record = []
	offset = 0
	time_offset = 0.0
	i = 0
	time_range = 0.0
	# num_record = int(np.ceil(duration/SEG_DURATION))
	while  time_range < duration/MS_IN_S:
		# print time_trace[starting_time_idx + i + offset]
		tp_record.append(tp_trace[starting_time_idx + i + offset])
		time_record.append(time_trace[starting_time_idx + i + offset] + time_offset)
		i += 1
		if starting_time_idx + i + offset >= len(tp_trace):
			offset -= len(tp_trace)
			time_offset += time_trace[-1]
		time_range = time_trace[starting_time_idx + i + offset] + time_offset - start_time

	return tp_record, time_record

def t_main():
	np.random.seed(RANDOM_SEED)
	if not os.path.isdir(LOG_FILE_DIR):
		os.makedirs(LOG_FILE_DIR)
	cooked_times, cooked_bws, cooked_names = load.new_loadBandwidth(DATA_DIR)

	for i in range(len(cooked_times)):
		cooked_time = cooked_times[i]
		cooked_bw = cooked_bws[i]
		cooked_name = cooked_names[i]

		# Trick here. For the initial bandwidth, directly get the first 5 value
		mpc_tp_rec = [INIT_BW] * MPC_STEP
		mpc_tp_pred = []

		player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
											seg_duration=SEG_DURATION,
											start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
											randomSeed=RANDOM_SEED)
		server = live_server.Live_Server(seg_duration=SEG_DURATION, 
											start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)

		initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
		print initial_delay, cooked_name
		log_path = LOG_FILE + '_' + cooked_name
		log_file = open(log_path, 'wb')

		starting_time = server.get_time()	# Server starting time
		starting_time_idx = player.get_time_idx()
		buffer_length = 0.0
		r_batch = []
		last_bit_rate = -1

		for i in range(TEST_DURATION):
			# print "Current index: ", i
			mpc_tp_pred = mpc.predict_mpc_tp(mpc_tp_rec)
			bit_rate_seq, opt_reward = mpc.mpc_find_action_seg([mpc_tp_pred, 0, player.get_real_time(), player.get_playing_time(), server.get_time(), \
									 player.get_buffer_length(), player.get_state(), last_bit_rate, 0.0, []])
			bit_rate = bit_rate_seq[0]
			# print "Bitrate is: ", bit_rate_seq, " and reward is: ", opt_reward
			# last_bit_rate = bit_rate
			# bit_rate = upper_actions[i]		# Get optimal actions
			# action_reward = 0.0				# Total reward is for all chunks within on segment

			download_seg_info = server.get_next_delivery()
			# print "seg info is " + str(download_seg_info)
			download_seg_idx = download_seg_info[0]
			download_seg_size = download_seg_info[1][bit_rate]
			# If sync happen, and just break
			if download_seg_idx > TEST_DURATION:
				break
			server_wait_time = 0.0
			sync = 0
			missing_count = 0
			real_seg_size, download_duration, freezing, time_out, player_state = player.fetch(download_seg_size, download_seg_idx)
			buffer_length = player.get_buffer_length()
			server_time = server.update(download_duration)

			# assert not time_out	# In MPC, current assume there is no timeout
			if not time_out:
				# server.chunks.pop(0)
				server.clean_next_delivery()
				sync = player.check_resync(server_time)
			else:
				assert player.get_state() == 0
				assert np.round(player.buffer, 3) == 0.0
				# Pay attention here, how time out influence next reward, the smoothness
				# Bit_rate will recalculated later, this is for reward calculation
				bit_rate = 0
				sync = 1
			# Disable sync for current situation
			if sync:
				# print "Should not happen!!!!!"
				# break	# No resync here
				print "Sync happen"
				# To sync player, enter start up phase, buffer becomes zero
				sync_time, missing_count = server.sync_encoding_buffer()
				player.sync_playing(sync_time)
				buffer_length = player.get_buffer_length()

			latency = server.get_time() - player.get_playing_time()
			# print "latency is: ", latency/MS_IN_S
			player_state = player.get_state()
			log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
			if last_bit_rate == -1:
				log_last_bit_rate = log_bit_rate
			else:
				log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
			last_bit_rate = bit_rate
			# print(log_bit_rate, log_last_bit_rate)
			reward = ACTION_REWARD * log_bit_rate  \
					- REBUF_PENALTY * freezing / MS_IN_S \
					- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
					- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) \
					- MISSING_PENALTY * missing_count

			if server.check_segs_empty():
				# print "Enter wait"
				server_wait_time = server.wait()
				# print " Has to wait: ", server_wait_time
				assert server_wait_time > 0.0
				assert server_wait_time < SEG_DURATION
				# print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
				player.wait(server_wait_time)
				# print "After wait, player: ", player.get_playing_time(), player.get_real_time()
				buffer_length = player.get_buffer_length()

			# print "After wait, ", server.get_time() - (seg_idx + 1) * SEG_DURATION
			mpc_tp_rec = mpc.update_mpc_rec(mpc_tp_rec, real_seg_size/download_duration * KB_IN_MB)

			server.generate_next_delivery()			
			# print(action_reward)
			r_batch.append(reward)
			log_file.write(	str(server.get_time()) + '\t' +
						    str(BITRATE[bit_rate]) + '\t' +
							str(buffer_length) + '\t' +
							str(freezing) + '\t' +
							str(time_out) + '\t' +
							str(server_wait_time) + '\t' +
						    str(sync) + '\t' +
						    str(latency) + '\t' +
						    str(player.get_state()) + '\t' +
						    str(int(bit_rate/len(BITRATE))) + '\t' +						    
							str(reward) + '\n')
			log_file.flush()
			reward = 0.0

		# need to modify
		time_duration = server.get_time() - starting_time
		tp_record, time_record = new_record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
		# print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
		print "Entire reward is:", np.sum(r_batch)
		log_file.write('\t'.join(str(tp) for tp in tp_record))
		log_file.write('\n')
		log_file.write('\t'.join(str(time) for time in time_record))
		# log_file.write('\n' + str(IF_NEW))
		log_file.write('\n' + str(starting_time))
		log_file.write('\n')
		log_file.close()


def main():
	np.random.seed(RANDOM_SEED)

	if not os.path.exists(LOG_FILE_DIR):
		os.makedirs(LOG_FILE_DIR)

	if not IF_NEW:
		cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + TRACE_NAME)
	else:
		cooked_time, cooked_bw = load.new_load_single_trace(DATA_DIR + TRACE_NAME)
	
	# Trick here. For the initial bandwidth, directly get the first 5 value
	mpc_tp_rec = [INIT_BW] * MPC_STEP
	mpc_tp_pred = []

	player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
										seg_duration=SEG_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=RANDOM_SEED)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, 
										start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)

	initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
	print initial_delay
	log_path = LOG_FILE + '_' + TRACE_NAME
	log_file = open(log_path, 'wb')

	starting_time = server.get_time()	# Server starting time
	starting_time_idx = player.get_time_idx()
	buffer_length = 0.0
	r_batch = []
	last_bit_rate = -1

	for i in range(TEST_DURATION):
		print "Current index: ", i
		mpc_tp_pred = mpc.predict_mpc_tp(mpc_tp_rec)
		bit_rate_seq, opt_reward = mpc.mpc_find_action_seg([mpc_tp_pred, 0, player.get_real_time(), player.get_playing_time(), server.get_time(), \
								 player.get_buffer_length(), player.get_state(), last_bit_rate, 0.0, []])
		bit_rate = bit_rate_seq[0]
		print "Bitrate is: ", bit_rate_seq, " and reward is: ", opt_reward
		# last_bit_rate = bit_rate
		# bit_rate = upper_actions[i]		# Get optimal actions
		# action_reward = 0.0				# Total reward is for all chunks within on segment

		download_seg_info = server.get_next_delivery()
		print "seg info is " + str(download_seg_info)
		download_seg_idx = download_seg_info[0]
		download_seg_size = download_seg_info[1][bit_rate]
		server_wait_time = 0.0
		sync = 0
		missing_count = 0
		if IF_NEW:
			if download_seg_idx >= TEST_DURATION:
				break
		real_seg_size, download_duration, freezing, time_out, player_state = player.fetch(download_seg_size, download_seg_idx)
		buffer_length = player.get_buffer_length()
		server_time = server.update(download_duration)

		assert not time_out	# In MPC, current assume there is no timeout
		if not time_out:
			# server.chunks.pop(0)
			server.clean_next_delivery()
			sync = player.check_resync(server_time)
		else:
			assert player.get_state() == 0
			assert np.round(player.buffer, 3) == 0.0
			# Pay attention here, how time out influence next reward, the smoothness
			# Bit_rate will recalculated later, this is for reward calculation
			bit_rate = 0
			if IF_NEW:
				sync = 1
		# Disable sync for current situation
		if sync:
			if not IF_NEW:
				print "Should not happen!!!!!"
				break	# No resync here
			# To sync player, enter start up phase, buffer becomes zero
			sync_time, missing_count = server.sync_encoding_buffer()
			player.sync_playing(sync_time)
			buffer_length = player.get_buffer_length()

		latency = server.get_time() - player.get_playing_time()
		# print "latency is: ", latency/MS_IN_S
		player_state = player.get_state()
		log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
		if last_bit_rate == -1:
			log_last_bit_rate = log_bit_rate
		else:
			log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
		last_bit_rate = bit_rate
		# print(log_bit_rate, log_last_bit_rate)
		reward = ACTION_REWARD * log_bit_rate  \
				- REBUF_PENALTY * freezing / MS_IN_S \
				- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
				- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) \
				- MISSING_PENALTY * missing_count
				# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
		# print(reward)
		# action_reward += reward

		# chech whether need to wait, using number of available segs
		if server.check_segs_empty():
			# print "Enter wait"
			server_wait_time = server.wait()
			# print " Has to wait: ", server_wait_time
			assert server_wait_time > 0.0
			assert server_wait_time < SEG_DURATION
			# print "Before wait, player: ", player.get_playing_time(), player.get_real_time()
			player.wait(server_wait_time)
			# print "After wait, player: ", player.get_playing_time(), player.get_real_time()
			buffer_length = player.get_buffer_length()

		# print "After wait, ", server.get_time() - (seg_idx + 1) * SEG_DURATION
		mpc_tp_rec = mpc.update_mpc_rec(mpc_tp_rec, real_seg_size/download_duration * KB_IN_MB)

		server.generate_next_delivery()			
		if sync and not IF_NEW:
			# Process sync
			print "Should not happen!!!!!!"
			pass
		else:
			# print(action_reward)
			r_batch.append(reward)
			log_file.write(	str(server.get_time()) + '\t' +
						    str(BITRATE[bit_rate]) + '\t' +
							str(buffer_length) + '\t' +
							str(freezing) + '\t' +
							str(time_out) + '\t' +
							str(server_wait_time) + '\t' +
						    str(sync) + '\t' +
						    str(latency) + '\t' +
						    str(player.get_state()) + '\t' +
						    str(int(bit_rate/len(BITRATE))) + '\t' +						    
							str(reward) + '\n')
			log_file.flush()
			reward = 0.0

	# need to modify
	time_duration = server.get_time() - starting_time
	if not IF_NEW:
		tp_record, time_record = record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
	else:
		tp_record, time_record = new_record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
	# print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
	print "Entire reward is:", np.sum(r_batch)
	log_file.write('\t'.join(str(tp) for tp in tp_record))
	log_file.write('\n')
	log_file.write('\t'.join(str(time) for time in time_record))
	# log_file.write('\n' + str(IF_NEW))
	log_file.write('\n' + str(starting_time))
	log_file.write('\n')
	log_file.close()

if __name__ == '__main__':
	if IF_ALL_TESTING:
		assert IF_NEW == 1
		t_main()
	else:
		main()
