import os
import logging
import numpy as np
import live_player_chunk as live_player
import live_server_chunk as live_server
import load
import mpc_solver_chunk as mpc
import math
import new_iLQR as iLQR
# import iLQR

IF_NEW = 1
IF_ALL_TESTING = 1
COMPARE_ILQR_VERSION = 0
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
CHUNK_DURATION = 200.0
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION

# Initial buffer length on server side
SERVER_START_UP_TH = 4000.0											# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0											# Single time freezing time upper bound
USER_LATENCY_TOL = SERVER_START_UP_TH + USER_FREEZING_TOL			# Accumulate latency upperbound

ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
REBUF_PENALTY = 6.0		# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 4.0 * CHUNK_SEG_RATIO 
MISSING_PENALTY = 6.0 * CHUNK_SEG_RATIO 		# not included
CONST = 6.0
X_RATIO = 1.0

# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1

TEST_DURATION = 100				# Number of testing <===================== Change length here
RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others
MPC_STEP = 5
# bitrate number is 6, no bin
BUFFER_AVE_LEN = 5

if not IF_NEW:
	DATA_DIR = '../../bw_traces_test/cooked_test_traces/'
	TRACE_NAME = '30+-10ms_loss0.5_1_2.txt'
else:
	DATA_DIR = '../../new_traces/test_sim_traces/'
	TRACE_NAME = 'norway_car_1'

if not IF_ALL_TESTING:
	LOG_FILE_DIR = './test_results'
	LOG_FILE = LOG_FILE_DIR + '/MPC_iLQR_CHUNK_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'
else:
	if IF_NEW:
		LOG_FILE_DIR = './all_test_results'
		LOG_FILE = LOG_FILE_DIR + '/MPC_iLQR_CHUNK_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'
		ALL_TESTING_DIR = '../../benchmark_compare/all_results/'
		ALL_TESTING_FILE = ALL_TESTING_DIR + 'MPC_iLQR_CHUNK_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's.txt'
	else:
		LOG_FILE_DIR = './all_test_results_old'
		LOG_FILE = LOG_FILE_DIR + '/MPCCHUNK_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'
		ALL_TESTING_DIR = '../../benchmark_compare/all_results_old/'
		ALL_TESTING_FILE = ALL_TESTING_DIR + 'MPC_iLQR_CHUNK_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's.txt'

def ReLU(x):
	return x * (x > 0)

def lat_penalty(x):
	return 1.0/(1+math.exp(CONST-X_RATIO*x)) - 1.0/(1+math.exp(CONST))

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
	start_time = time_trace[starting_time_idx]
	tp_record = []
	time_record = []
	offset = 0
	time_offset = 0.0
	i = 0
	time_range = 0.0
	# num_record = int(np.ceil(duration/SEG_DURATION))
	while  time_range < duration/MS_IN_S:
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
	if not os.path.exists(LOG_FILE_DIR):
		os.makedirs(LOG_FILE_DIR)

	if not os.path.isdir(ALL_TESTING_DIR):
		os.makedirs(ALL_TESTING_DIR)
	all_testing_log = open(ALL_TESTING_FILE, 'w')

	if IF_NEW:
		cooked_times, cooked_bws, cooked_names = load.new_loadBandwidth(DATA_DIR)
	else:
		cooked_times, cooked_bws, cooked_names = load.loadBandwidth(DATA_DIR)

	iLQR_solver = iLQR.iLQR_solver()
	iLQR_solver.set_step()

	for i in range(len(cooked_times)):
		cooked_time = cooked_times[i]
		cooked_bw = cooked_bws[i]
		cooked_name = cooked_names[i]

		# Trick here. For the initial bandwidth, directly get the first 5 value
		mpc_tp_rec = [INIT_BW] * MPC_STEP
		mpc_tp_pred = []

		player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
											seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
											start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
											randomSeed=RANDOM_SEED)
		server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
											start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)
		initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
		print(initial_delay, cooked_name)
		log_path = LOG_FILE + '_' + cooked_name
		log_file = open(log_path, 'w')

		init = 1
		starting_time = server.get_time()	# Server starting time
		starting_time_idx = player.get_time_idx()
		buffer_length = 0.0
		r_batch = []
		f_batch = []
		a_batch = []
		c_batch = []
		l_batch = []
		buffer_his = []
		last_bit_rate = -1
		for i in range(TEST_DURATION):
			print("Current index: ", i)
			if init: 
				if CHUNK_IN_SEG == 5:
					ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
				else:
					ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
				
				server.set_ratio(ratio)
				server.init_encoding()
				init = 0
			mpc_tp_pred = mpc.predict_mpc_tp(mpc_tp_rec)
			buffer_his.append(player.get_buffer_length()/MS_IN_S)
			if len(buffer_his) >= BUFFER_AVE_LEN:
				del buffer_his[0]
			# Method 1:
			# bit_rate_seq, opt_reward = mpc.mpc_find_action_chunk([mpc_tp_pred, 0, player.get_real_time(), player.get_playing_time(), server.get_time(), \
			# 						 player.get_buffer_length(), player.get_state(), last_bit_rate, 0.0, [], ratio])
			# bit_rate = bit_rate_seq[0]
			# print("Bitrate is: ", bit_rate_seq, " and reward is: ", opt_reward)

			# Method 2: iLQR
			if player.get_buffer_length() == 0:
				bit_rate = 0
			else:
				latency = server.get_time() - player.get_playing_time()
				iLQR_solver.set_target_buff(np.mean(buffer_his))
				iLQR_solver.set_bu(latency)
				iLQR_solver.set_predicted_bw_rtt(mpc_tp_pred)
				if COMPARE_ILQR_VERSION == 1:
					ilqr_rates = []
					for iLQR_v in range(4):
						if last_bit_rate == -1:
							iLQR_solver.set_x0(player.get_buffer_length())
						else:
							iLQR_solver.set_x0(player.get_buffer_length(), BITRATE[last_bit_rate])
							if iLQR_v == 0:
								iLQR_solver.generate_initial_x(min(mpc_tp_pred))
							elif iLQR_v == 1:
								iLQR_solver.generate_initial_x(mpc_tp_pred[0])
							elif iLQR_v == 2:
								iLQR_solver.generate_initial_x(np.mean(mpc_tp_pred))
							elif iLQR_v == 3:
								iLQR_solver.generate_initial_x_trace(mpc_tp_pred)
							bit_rate = iLQR_solver.iterate_LQR()
							ilqr_rates.append(iLQR_solver.get_rates())
					for ilqr_res in ilqr_rates:
						print(ilqr_res)
				else:
					if last_bit_rate == -1:
						iLQR_solver.set_x0(player.get_buffer_length())
					else:
						iLQR_solver.set_x0(player.get_buffer_length(), BITRATE[last_bit_rate])
						iLQR_solver.generate_initial_x(mpc_tp_pred[0])
						bit_rate = iLQR_solver.iterate_LQR()
						if iLQR_solver.checking():
							bit_rate = iLQR_solver.nan_index(mpc_tp_pred[0]/KB_IN_MB)

			c_batch.append(np.abs(BITRATE[bit_rate] - BITRATE[last_bit_rate]))
			# bit_rate = upper_actions[i]		# Get optimal actions
			action_reward = 0.0				# Total reward is for all chunks within on segment
			take_action = 1
			current_mpc_tp = 0.0
			seg_freezing = 0.0
			seg_wait = 0.0

			while True:  # serve video forever
				download_chunk_info = server.get_next_delivery()
				download_seg_idx = download_chunk_info[0]
				download_chunk_idx = download_chunk_info[1]
				download_chunk_end_idx = download_chunk_info[2]
				download_chunk_size = download_chunk_info[3][bit_rate]		# Might be several chunks
				chunk_number = download_chunk_end_idx - download_chunk_idx + 1
				assert chunk_number == 1
				server_wait_time = 0.0
				sync = 0
				missing_count = 0
				if download_seg_idx >= TEST_DURATION:
					break
				real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(download_chunk_size, 
																		download_seg_idx, download_chunk_idx, take_action, chunk_number)
				take_action = 0
				current_mpc_tp += chunk_number/CHUNK_IN_SEG * real_chunk_size / download_duration
				buffer_length = player.get_buffer_length()
				seg_freezing += freezing
				server_time = server.update(download_duration)
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
					# To sync player, enter start up phase, buffer becomes zero
					sync_time, missing_count = server.sync_encoding_buffer()
					player.sync_playing(sync_time)
					buffer_length = player.get_buffer_length()

				latency = server.get_time() - player.get_playing_time()
				player_state = player.get_state()

				log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
				if last_bit_rate == -1:
					log_last_bit_rate = log_bit_rate
				else:
					log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
				last_bit_rate = bit_rate	# Do no move this term. This is for chunk continuous calcualtion

				reward = ACTION_REWARD * log_bit_rate * chunk_number \
						- REBUF_PENALTY * freezing / MS_IN_S \
						- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
						- LONG_DELAY_PENALTY * lat_penalty(latency/ MS_IN_S) * chunk_number \
						- MISSING_PENALTY * missing_count
						# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
				# print(reward)
				action_reward += reward

				# chech whether need to wait, using number of available segs
				if server.check_chunks_empty():
					server_wait_time = server.wait()
					seg_wait += server_wait_time
					assert server_wait_time > 0.0
					assert server_wait_time < CHUNK_DURATION
					player.wait(server_wait_time)
					buffer_length = player.get_buffer_length()

				if CHUNK_IN_SEG == 5:
					ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
				else:
					ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
				server.set_ratio(ratio)
				server.generate_next_delivery()
				next_chunk_idx = server.get_next_delivery()[1]
				if next_chunk_idx == 0 or sync:
					# Record state and get reward
					take_action = 1
					mpc_tp_rec = mpc.update_mpc_rec(mpc_tp_rec, current_mpc_tp * KB_IN_MB)
					r_batch.append(action_reward)
					f_batch.append(seg_freezing)
					a_batch.append(BITRATE[bit_rate])
					l_batch.append(latency)

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
									str(action_reward) + '\n')
					log_file.flush()
					action_reward = 0.0
					break

		# need to modify
		time_duration = server.get_time() - starting_time
		tp_record, time_record = new_record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
		print("Entire reward is:", np.sum(r_batch))
		log_file.write('\t'.join(str(tp) for tp in tp_record))
		log_file.write('\n')
		log_file.write('\t'.join(str(time) for time in time_record))
		# log_file.write('\n' + str(IF_NEW))
		log_file.write('\n' + str(starting_time))
		log_file.write('\n')
		log_file.close()

		#write to all testing
		all_testing_log.write(cooked_name + '\t')
		all_testing_log.write(str(np.sum(r_batch)) + '\t')
		all_testing_log.write(str(np.mean(a_batch)) + '\t')
		all_testing_log.write(str(np.sum(f_batch)) + '\t')
		all_testing_log.write(str(np.mean(c_batch)) + '\t')
		all_testing_log.write(str(np.mean(l_batch)) + '\t')

		all_testing_log.write('\n')
	all_testing_log.close()

def main():
	np.random.seed(RANDOM_SEED)

	if not os.path.exists(LOG_FILE_DIR):
		os.makedirs(LOG_FILE_DIR)

	if not IF_NEW:
		cooked_time, cooked_bw = load.load_single_trace(DATA_DIR + TRACE_NAME)
	else:
		cooked_time, cooked_bw = load.new_load_single_trace(DATA_DIR + TRACE_NAME)

	iLQR_solver = iLQR.iLQR_solver()
	iLQR_solver.set_step()
	# Trick here. For the initial bandwidth, directly get the first 5 value
	mpc_tp_rec = [INIT_BW] * MPC_STEP
	mpc_tp_pred = []

	player = live_player.Live_Player(time_trace=cooked_time, throughput_trace=cooked_bw, 
										seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=RANDOM_SEED)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, 
										start_up_th=SERVER_START_UP_TH, randomSeed=RANDOM_SEED)

	initial_delay = server.get_time() - player.get_playing_time()	# This initial delay, cannot be reduced, all latency is calculated based on this
	print(initial_delay)
	log_path = LOG_FILE + '_' + TRACE_NAME
	log_file = open(log_path, 'w')

	init = 1
	starting_time = server.get_time()	# Server starting time
	starting_time_idx = player.get_time_idx()
	buffer_length = 0.0
	r_batch = []
	last_bit_rate = -1
	buffer_his = []

	for i in range(TEST_DURATION):
		print("Current index: ", i)
		if init: 
			if CHUNK_IN_SEG == 5:
				ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
			else:
				ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
			
			server.set_ratio(ratio)
			server.init_encoding()
			init = 0
		mpc_tp_pred = mpc.predict_mpc_tp(mpc_tp_rec)
		buffer_his.append(player.get_buffer_length()/MS_IN_S)
		if len(buffer_his) >= BUFFER_AVE_LEN:
			del buffer_his[0]
		# Method 1: greedy search
		# bit_rate_seq, opt_reward = mpc.mpc_find_action_chunk([mpc_tp_pred, 0, player.get_real_time(), player.get_playing_time(), server.get_time(), \
		# 						 player.get_buffer_length(), player.get_state(), last_bit_rate, 0.0, [], ratio])
		# bit_rate = bit_rate_seq[0]
		# print("Bitrate is: ", bit_rate_seq, " and reward is: ", opt_reward)

		# Method 2: iLQR
		if player.get_buffer_length() == 0:
			bit_rate = 0
		else:
			latency = server.get_time() - player.get_playing_time()
			iLQR_solver.set_target_buff(np.mean(buffer_his))
			iLQR_solver.set_bu(latency)
			iLQR_solver.set_predicted_bw_rtt(mpc_tp_pred)
			if COMPARE_ILQR_VERSION == 1:
				ilqr_rates = []
				for iLQR_v in range(4):
					if last_bit_rate == -1:
						iLQR_solver.set_x0(player.get_buffer_length())
					else:
						iLQR_solver.set_x0(player.get_buffer_length(), BITRATE[last_bit_rate])
						if iLQR_v == 0:
							iLQR_solver.generate_initial_x(min(mpc_tp_pred))
						elif iLQR_v == 1:
							iLQR_solver.generate_initial_x(mpc_tp_pred[0])
						elif iLQR_v == 2:
							iLQR_solver.generate_initial_x(np.mean(mpc_tp_pred))
						elif iLQR_v == 3:
							iLQR_solver.generate_initial_x_trace(mpc_tp_pred)
						bit_rate = iLQR_solver.iterate_LQR()
						ilqr_rates.append(iLQR_solver.get_rates())
				for ilqr_res in ilqr_rates:
					print(ilqr_res)
			else:
				if last_bit_rate == -1:
					iLQR_solver.set_x0(player.get_buffer_length())
				else:
					iLQR_solver.set_x0(player.get_buffer_length(), BITRATE[last_bit_rate])
					iLQR_solver.generate_initial_x(mpc_tp_pred[0])
					bit_rate = iLQR_solver.iterate_LQR()
					if iLQR_solver.checking():
						bit_rate = iLQR_solver.nan_index(mpc_tp_pred[0]/KB_IN_MB)

		# bit_rate = upper_actions[i]		# Get optimal actions
		action_reward = 0.0				# Total reward is for all chunks within on segment
		take_action = 1
		current_mpc_tp = 0.0
		seg_freezing = 0.0
		seg_wait = 0.0

		while True:  # serve video forever
			download_chunk_info = server.get_next_delivery()
			download_seg_idx = download_chunk_info[0]
			download_chunk_idx = download_chunk_info[1]
			download_chunk_end_idx = download_chunk_info[2]
			download_chunk_size = download_chunk_info[3][bit_rate]		# Might be several chunks
			chunk_number = download_chunk_end_idx - download_chunk_idx + 1
			server_wait_time = 0.0
			sync = 0
			missing_count = 0
			if IF_NEW:
				if download_seg_idx >= TEST_DURATION:
					break
			real_chunk_size, download_duration, freezing, time_out, player_state = player.fetch(download_chunk_size, 
																	download_seg_idx, download_chunk_idx, take_action, chunk_number)
			take_action = 0
			current_mpc_tp += chunk_number/CHUNK_IN_SEG * real_chunk_size / download_duration
			buffer_length = player.get_buffer_length()
			seg_freezing += freezing
			server_time = server.update(download_duration)
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
					print("Should not happen!")
					break	# No resync here
				# To sync player, enter start up phase, buffer becomes zero
				sync_time, missing_count = server.sync_encoding_buffer()
				player.sync_playing(sync_time)
				buffer_length = player.get_buffer_length()

			latency = server.get_time() - player.get_playing_time()
			# print("latency is: ", latency/MS_IN_S)
			player_state = player.get_state()

			log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
			if last_bit_rate == -1:
				log_last_bit_rate = log_bit_rate
			else:
				log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
			last_bit_rate = bit_rate	# Do no move this term. This is for chunk continuous calcualtion

			reward = ACTION_REWARD * log_bit_rate * chunk_number \
				- REBUF_PENALTY * freezing / MS_IN_S \
				- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
				- LONG_DELAY_PENALTY * lat_penalty(latency/ MS_IN_S) * chunk_number \
				- MISSING_PENALTY * missing_count
					# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S
			# print(reward)
			action_reward += reward

			# chech whether need to wait, using number of available segs
			if server.check_chunks_empty():
				server_wait_time = server.wait()
				seg_wait += server_wait_time
				assert server_wait_time > 0.0
				assert server_wait_time < CHUNK_DURATION
				player.wait(server_wait_time)
				buffer_length = player.get_buffer_length()

			if CHUNK_IN_SEG == 5:
				ratio = np.random.uniform(RATIO_LOW_5, RATIO_HIGH_5)
			else:
				ratio = np.random.uniform(RATIO_LOW_2, RATIO_HIGH_2)
			server.set_ratio(ratio)
			server.generate_next_delivery()
			next_chunk_idx = server.get_next_delivery()[1]
			if next_chunk_idx == 0 or sync:
				# Record state and get reward
				if sync and not IF_NEW:
					# Process sync
					pass
				else:
					take_action = 1
					mpc_tp_rec = mpc.update_mpc_rec(mpc_tp_rec, current_mpc_tp * KB_IN_MB)
					r_batch.append(action_reward)

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
									str(action_reward) + '\n')
					log_file.flush()
					action_reward = 0.0
					break

	# need to modify
	time_duration = server.get_time() - starting_time
	if not IF_NEW:
		tp_record, time_record = record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
	else:
		tp_record, time_record = new_record_tp(player.get_throughput_trace(), player.get_time_trace(), starting_time_idx, time_duration + buffer_length) 
	# tp_record = record_tp(player.get_throughput_trace(), starting_time_idx, time_duration) 
	print(starting_time_idx, TRACE_NAME, len(player.get_throughput_trace()), player.get_time_idx(), len(tp_record), np.sum(r_batch))
	log_file.write('\t'.join(str(tp) for tp in tp_record))
	log_file.write('\n')
	log_file.write('\t'.join(str(time) for time in time_record))
	# log_file.write('\n' + str(IF_NEW))
	log_file.write('\n' + str(starting_time))
	log_file.write('\n')
	log_file.close()

if __name__ == '__main__':
	if IF_ALL_TESTING:
		# assert IF_NEW == 1
		t_main()
	else:
		main()
