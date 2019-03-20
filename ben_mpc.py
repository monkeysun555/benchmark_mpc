import os
import logging
import numpy as np
import live_player
import live_server
import load

MAX_LIVE_LEN = 200
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]  # 5 actions
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]

BITRATE_LOW_NOISE = 0.95
BITRATE_HIGH_NOISE = 1.05
# MABW_DECAY = 0.9
BW_RATIO = 0.5

MS_IN_S = 1000.0
KB_IN_MB = 1000.0   # in ms

SEG_DURATION = 2000.0

SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
USER_FREEZING_TOL = 3000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * FRAG_DURATION
USER_LATENCY_TOL = TARGET_LATENCY + 3000.0


DEFAULT_ACTION = 0	# lowest bitrate of ET
ACTION_REWARD = 1.0
REBUF_PENALTY = 10.0	# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 1.0  
LONG_DELAY_PENALTY_BASE = 1.2	# for second
# MISSING_PENALTY = 2.0	# not included
# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1

RANDOM_SEED = 1
RAND_RANGE = 1000

TEST_TRACE_NUM = 4
LOG_FILE_DIR = './test_results/naive/'+ str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'
LOG_FILE = LOG_FILE_DIR + '/naive_' + str(int(SERVER_START_UP_TH/MS_IN_S)) + 's'
if not os.path.isdir(LOG_FILE_DIR):
	os.makedirs(LOG_FILE_DIR)

TEST_TRACES = '../test_traces/'

def ReLU(x):
	return x * (x > 0)

def record_tp(tp_trace, starting_time_idx, duration):
	tp_record = []
	offset = 0
	num_record = int(np.ceil(duration/FRAG_DURATION))
	for i in range(num_record):
		if starting_time_idx + i + offset >= len(tp_trace):
			offset = -len(tp_trace)
		tp_record.append(tp_trace[starting_time_idx + i + offset])
	return tp_record

def estimate_tp(bw_history):
	weight_sum = np.sum([(i+1)*bw_history[i] for i in range(len(bw_history))])
	return weight_sum/np.sum(range(len(bw_history)+1))

def choose_rate(est_bw):
	true_bw = BW_RATIO*est_bw
	for i in reversed(range(len(BITRATE))):
		if BITRATE[i] <= true_bw*KB_IN_MB:
			return i
	return 0

def main():

	np.random.seed(RANDOM_SEED)
	all_cooked_time, all_cooked_bw, all_file_names = load.loadBandwidth(TEST_TRACES)

	player = live_player.Live_Player(time_traces=all_cooked_time, throughput_traces=all_cooked_bw, 
										seg_duration=SEG_DURATION,
										start_up_th=USER_START_UP_TH, freezing_tol=USER_FREEZING_TOL, latency_tol = USER_LATENCY_TOL,
										randomSeed=agent_id)
	server = live_server.Live_Server(seg_duration=SEG_DURATION, start_up_th=SERVER_START_UP_TH)

	log_path = LOG_FILE + '_' + all_file_names[player.trace_idx]
	log_file = open(log_path, 'wb')

	action_num = DEFAULT_ACTION	# 0
	last_bit_rate = action_num
	bit_rate = action_num

	reward = 0.0		# Total reward is for all chunks within on segment
	latency = 0.0
	video_count = 0
	starting_time = server.get_time()
	starting_time_idx = player.get_time_idx()
	r_batch = []
	est_bw = 0.0

	while True:  # serve video forever
		assert len(server.chunks) >= 1
		download_seg_info = server.get_next_delivery()
		download_seg_idx = download_seg_info[0]
		download_seg_size = download_seg_info[1]
		server_wait_time = 0.0
		sync = 0
		missing_count = 0
		real_seg_size, download_duration, freezing, time_out, player_state = player.fetch(bit_rate, download_seg_size, 
																		download_seg_idx)
		player.update_bw(real_chunk_size/download_duration)
		past_time = download_duration	#including freezing time
		buffer_length = player.get_buffer_len()
		# print(player.playing_time)
		server.update(past_time)
		server_time = server.get_time()

		# print(freezing, time_out)		
		if not time_out:
			server.process_delivery()
			sync = player.check_resync(server_time)
		else:
			assert player.get_state() == 0
			assert np.round(player.get_buffer_len(), 3) == 0.0
			# Pay attention here, how time out influence next reward, the smoothness
			# Bit_rate will recalculated later, this is for reward calculation
			bit_rate = 0
			sync = 1
		if sync:
			# To sync player, enter start up phase, buffer becomes zero
			sync_time, _ = server.sync_encoding_buffer()
			player.sync_playing(sync_time)
			buffer_length = player.get_buffer_len()

		latency = server.get_time() - player.get_time()
		player_state = player.get_state()

		log_bit_rate = np.log(BITRATE[bit_rate] / BITRATE[0])
		log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
		last_bit_rate = bit_rate
		# print(log_bit_rate, log_last_bit_rate)
		reward = ACTION_REWARD * log_bit_rate \
				- REBUF_PENALTY * freezing / MS_IN_S \
				- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
				- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) \
				- MISSING_PENALTY * missing_count
				# - UNNORMAL_PLAYING_PENALTY*(playing_speed-NORMAL_PLAYING)*download_duration/MS_IN_S

		# chech whether need to wait, using number of available segs
		if server.check_encoding_buff() == 0:
			server_wait_time = server.wait()
			assert server_wait_time > 0.0
			assert server_wait_time < SEG_DURATION
			player.wait(server_wait_time)
			buffer_length = player.get_buffer_len()
		server.generate_next_delivery()


		r_batch.append(reward)
		reward = 0.0
		
		# Estimate bw and take action
		est_bw = estimate_tp(player.bw)
		# print(est_bw)
		action_num = choose_rate(est_bw)			#<===============using MPC
		# print(action_num)
		bit_rate = action_num

		# if action_num >= len(BITRATE):
		# 	playing_speed = FAST_PLAYING
		# else:
		# 	playing_speed = NORMAL_PLAYING

		log_file.write(	str(server.time) + '\t' +
					    str(BITRATE[last_bit_rate]) + '\t' +
						str(buffer_length) + '\t' +
						str(freezing) + '\t' +
						str(time_out) + '\t' +
						str(server_wait_time) + '\t' +
					    str(sync) + '\t' +
					    str(latency) + '\t' +
					    str(player.state) + '\t' +
					    str(int(action_num/len(BITRATE))) + '\t' +						    
						str(reward) + '\n')
		log_file.flush()

		if len(r_batch) >= MAX_LIVE_LEN:
			# need to modify
			time_duration = server.time - starting_time
			tp_record = record_tp(player.throughput_trace, starting_time_idx, time_duration) 
			print(starting_time_idx, all_file_names[player.trace_idx], len(player.throughput_trace), player.time_idx, len(tp_record), np.sum(r_batch))
			log_file.write('\t'.join(str(tp) for tp in tp_record))
			log_file.write('\n' + str(starting_time))
			log_file.write('\n')
			log_file.close()

			action_num = DEFAULT_ACTION	# 0
			last_bit_rate = action_num
			bit_rate = action_num
			# playing_speed = NORMAL_PLAYING
			video_count += 1

			if video_count >= TEST_TRACE_NUM:
				break

			player.test_reset(start_up_th=USER_START_UP_TH, random_seed=RANDOM_SEED + video_count)
			server.test_reset(start_up_th=SERVER_START_UP_TH)
			r_batch = []
			# Do not need to append state to s_batch as there is no iteration
			starting_time = server.time
			starting_time_idx = player.time_idx
			log_path = LOG_FILE + '_' + all_file_names[player.trace_idx]
			log_file = open(log_path, 'wb')
			take_action = 1

if __name__ == '__main__':
	main()
