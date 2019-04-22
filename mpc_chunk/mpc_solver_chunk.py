# import cplex
import numpy as np

BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]

RTT_LOW = 30.0
RTT_HIGH = 40.0
MS_IN_S = 1000.0
KB_IN_MB = 1000.0

SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
CHUNK_DURATION = 200.0
CHUNK_IN_SEG = SEG_DURATION/CHUNK_DURATION
CHUNK_SEG_RATIO = CHUNK_DURATION/SEG_DURATION
# Initial buffer length on server side
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
# how user will start playing video (user buffer)
USER_START_UP_TH = 2000.0
# set a target latency, then use fast playing to compensate
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
USER_FREEZING_TOL = 3000.0							# Single time freezing time upper bound
USER_LATENCY_TOL = TARGET_LATENCY + 3000.0			# Accumulate latency upperbound


DEFAULT_ACTION = 0			# lowest bitrate
ACTION_REWARD = 1.0 * CHUNK_SEG_RATIO	
REBUF_PENALTY = 10.0		# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 1.0 * CHUNK_SEG_RATIO 
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 2.0			# not included
# UNNORMAL_PLAYING_PENALTY = 1.0 * CHUNK_FRAG_RATIO
# FAST_PLAYING = 1.1		# For 1
# NORMAL_PLAYING = 1.0	# For 0
# SLOW_PLAYING = 0.9		# For -1

MPC_STEP = 5

def ReLU(x):
	return x * (x > 0)

def generate_chunks(ratio):
	current_seg_size = [[] for i in range(len(BITRATE))]
	encoding_coef = 1.0
	estimate_seg_size = [x * encoding_coef for x in BITRATE]

	if CHUNK_IN_SEG == 2:
	# Distribute size for chunks, currently, it should depend on chunk duration (200 or 500)
		# seg_ratio = [np.random.uniform(EST_LOW_NOISE*ratio, EST_HIGH_NOISE*ratio) for x in range(len(BITRATE))]
		for i in range(len(estimate_seg_size)):
			temp_aux_chunk_size = estimate_seg_size[i]/(1 + ratio)
			temp_ini_chunk_size = estimate_seg_size[i] - temp_aux_chunk_size
			current_seg_size[i].extend((temp_ini_chunk_size, temp_aux_chunk_size))
	# if 200ms, needs to be modified, not working
	elif CHUNK_IN_SEG == 5:
		for i in range(len(estimate_seg_size)):
			temp_ini_chunk_size = estimate_seg_size[i] * ratio / (1 + ratio)
			temp_aux_chunk_size = (estimate_seg_size[i] - temp_ini_chunk_size) / (CHUNK_IN_SEG - 1)
			temp_chunks_size = [temp_ini_chunk_size]
			temp_chunks_size.extend([temp_aux_chunk_size for _ in range(int(CHUNK_IN_SEG) - 1)])
			current_seg_size[i].extend(temp_chunks_size)
	else:
		assert 1 == 0
	return current_seg_size

# def mpc_solver(pred_tp, k, player_time, playback_time, server_time, buffer_length, state, last_bit_rate, pre_reward, seq, ratio):
def mpc_solver_chunk(mpc_input):
	# print mpc_input
	# Guarante there is available segment
	ratio = mpc_input[10]
	sys_state = []
	chunks_info = generate_chunks(ratio)
	# print "chunk info: ", chunks_info
	for i in range(len(BITRATE)):
		# print "Birte is: ", i
		current_chunks = chunks_info[i]
		pred_tp = mpc_input[0]
		k = mpc_input[1]
		player_time = mpc_input[2]
		playback_time = mpc_input[3]
		server_time = mpc_input[4]
		buffer_length = mpc_input[5]
		state = mpc_input[6]
		last_bit_rate = mpc_input[7]
		pre_reward = mpc_input[8]
		seq = mpc_input[9][:]
		buffer_to_accu = 0.0
		if state == 0:
			buffer_to_accu = USER_START_UP_TH - buffer_length
			assert buffer_to_accu > 0.0
		assert np.round(playback_time + buffer_length, 3) <= server_time - CHUNK_DURATION
		# print i 
		download_time = 0.0
		freezing = 0.0
		wait_time = 0.0
		current_reward = pre_reward
		missing_count = 0
		current_tp = pred_tp[k]
		downloaded_chunks = 0
		chunk_num = int(np.minimum((server_time - playback_time - buffer_length)/CHUNK_DURATION, CHUNK_IN_SEG - downloaded_chunks))
		# print "Init chunk num: ", chunk_num
		while True:
			if downloaded_chunks == 0:
				C_RTT = RTT_LOW
			else: 
				C_RTT = 0.0
			# print np.sum(current_chunks[downloaded_chunks:downloaded_chunks+chunk_num])
			# print current_tp
			download_time = np.sum(current_chunks[downloaded_chunks:downloaded_chunks+chunk_num])/current_tp * MS_IN_S + C_RTT
			# print "Download time: ", download_time
			if state == 0:
				freezing = download_time
				temp_buffer_to_accu = buffer_to_accu - CHUNK_DURATION * chunk_num
				if temp_buffer_to_accu == 0.0:
					state = 1
				assert playback_time == 0.0
				buffer_length += CHUNK_DURATION * chunk_num

			else:
				assert buffer_to_accu == 0.0
				freezing = np.maximum(0.0, download_time - buffer_length)
				playback_time += np.minimum(buffer_length, download_time)
				buffer_length = np.maximum(buffer_length - download_time, 0.0)
				buffer_length += CHUNK_DURATION * chunk_num
				if freezing > 0.0:
					# print "Freezing is: ", freezing
					# print "Buffer leng: ", buffer_length, " and chunk_num: ", chunk_num
					assert buffer_length == CHUNK_DURATION * chunk_num

			assert np.round(playback_time + buffer_length, 3) % CHUNK_DURATION == 0.0
			server_time += download_time
			player_time += download_time

			if server_time - buffer_length - playback_time < CHUNK_DURATION:
				wait_time = np.round(playback_time + buffer_length + CHUNK_DURATION, 3) - server_time
				# print "Wait time is: ", wait_time
			player_time +=  wait_time
			server_time +=  wait_time
			if state:
				assert buffer_length > wait_time
				buffer_length -= wait_time
				playback_time += wait_time

			latency = server_time - playback_time
			# print player_time, playback_time, server_time, freezing, buffer_length, state

			log_bit_rate = np.log(BITRATE[i] / BITRATE[0])
			if last_bit_rate == -1:
				assert chunk_num == CHUNK_IN_SEG
				log_last_bit_rate = log_bit_rate
			else:
				log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])

			# print "After fetching, buffer is: ", buffer_length, " download time: ", download_time
			# print "chunk num: ", chunk_num
			last_bit_rate = i
			current_reward += ACTION_REWARD * log_bit_rate * chunk_num \
							- REBUF_PENALTY * freezing / MS_IN_S \
							- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
							- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) * chunk_num \
							- MISSING_PENALTY * missing_count
			
			downloaded_chunks += chunk_num
			if downloaded_chunks == CHUNK_IN_SEG:
				break
			assert downloaded_chunks < CHUNK_IN_SEG
			chunk_num = int(np.minimum(np.round((server_time - playback_time - buffer_length)/CHUNK_DURATION, 3), CHUNK_IN_SEG - downloaded_chunks))
			# print server_time, playback_time, buffer_length
			assert chunk_num > 0



		seq.append(i)
		sys_state.append([pred_tp, k+1, player_time, playback_time, server_time, buffer_length, state, last_bit_rate, current_reward, seq, ratio])
		# print "done state is: ", sys_state
	k += 1
	if k == MPC_STEP:
		# print len(sys_state)
		return sys_state
	else:
		# print "<----------->"
		# print sys_state
		# print "<----------->"

		current_len = len(sys_state)
		# print current_len
		j = 0
		while j < current_len:
			# print sys_state, k, j
			# print "<?????>"
			sys_state.extend(mpc_solver_chunk(sys_state[0]))
			# print "<?------?>"
			sys_state.pop(0)
			# print sys_state
			# print len(sys_state)
			j += 1
	return sys_state

def mpc_find_opt_chunk(all_states):
	opt_reward = float("-inf")
	opt_action = -1
	for state in all_states:
		if state[8] > opt_reward:
			opt_action = state[9]
			opt_reward = state[8]
	return opt_action, opt_reward

def mpc_find_action_chunk(mpc_input):
	all_states = mpc_solver_chunk(mpc_input)
	return mpc_find_opt_chunk(all_states)


def update_mpc_rec(pred_tp, new_tp):
	curr_tp = np.roll(pred_tp, -1, axis=0)
	curr_tp[-1] = new_tp
	return curr_tp.tolist()

def predict_mpc_tp(updated_pred_tp):
	assert len(updated_pred_tp) == MPC_STEP
	combined_tp = updated_pred_tp[:]
	combined_tp.extend(np.zeros(MPC_STEP))
	for i in range(MPC_STEP):
		combined_tp[MPC_STEP + i] = harmonic_prediction(combined_tp[i:i+MPC_STEP])
	return combined_tp[-MPC_STEP:]

def harmonic_prediction(history):
	return len(history)/(np.sum([1/tp for tp in history]))
