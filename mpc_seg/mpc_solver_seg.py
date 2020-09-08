# import cplex
import numpy as np

K_P = 3

BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]

SEG_DURATION = 1000.0
USER_START_UP_TH = 2000.0
SERVER_START_UP_TH = 2000.0
TARGET_LATENCY = SERVER_START_UP_TH + 0.5 * SEG_DURATION
RTT_LOW = 30.0
RTT_HIGH = 40.0 
MS_IN_S = 1000.0
KB_IN_MB = 1000.0

DEFAULT_ACTION = 0			# lowest bitrate
ACTION_REWARD = 1.0 	
REBUF_PENALTY = 10.0		# for second
SMOOTH_PENALTY = 1.0
LONG_DELAY_PENALTY = 1.0  
LONG_DELAY_PENALTY_BASE = 1.2	# for second
MISSING_PENALTY = 2.0			# not included

MPC_STEP = 5

def ReLU(x):
	return x * (x > 0)

# def mpc_solver(pred_tp, k, player_time, playback_time, server_time, buffer_length, state, last_bit_rate, pre_reward, seq):
def mpc_solver_seg(mpc_input, pruning):
	# print mpc_input
	# Guarante there is available segment
	sys_state = []
	
	for i in range(len(BITRATE)):
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
		assert np.round(playback_time + buffer_length, 3) <= server_time - SEG_DURATION

		# print i 
		download_time = 0.0
		freezing = 0.0
		wait_time = 0.0
		current_reward = 0.0
		missing_count = 0
		current_tp = pred_tp[k]

		download_time = BITRATE[i]/current_tp * MS_IN_S + RTT_LOW
		if state == 0:
			freezing += download_time
			temp_buffer_to_accu = buffer_to_accu - SEG_DURATION
			if temp_buffer_to_accu == 0.0:
				state = 1
			# assert playback_time == 0.0
			buffer_length += SEG_DURATION

		else:
			assert buffer_to_accu == 0.0
			freezing += np.maximum(0.0, download_time - buffer_length)
			playback_time += np.minimum(buffer_length, download_time)
			buffer_length = np.maximum(buffer_length - download_time, 0.0)
			buffer_length += SEG_DURATION
			if freezing > 0.0:
				assert buffer_length == SEG_DURATION

		assert np.round(playback_time + buffer_length, 3) % MS_IN_S == 0.0
		server_time += download_time
		player_time += download_time

		if server_time - buffer_length - playback_time < SEG_DURATION:
			wait_time = np.round(playback_time + buffer_length + SEG_DURATION, 3) - server_time
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
			log_last_bit_rate = log_bit_rate
		else:
			log_last_bit_rate = np.log(BITRATE[last_bit_rate] / BITRATE[0])
		last_bit_rate = i
		seq.append(i)
		current_reward = pre_reward \
						+ ACTION_REWARD * log_bit_rate \
						- REBUF_PENALTY * freezing / MS_IN_S \
						- SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate) \
						- LONG_DELAY_PENALTY*(LONG_DELAY_PENALTY_BASE**(ReLU(latency-TARGET_LATENCY)/ MS_IN_S)-1) \
						- MISSING_PENALTY * missing_count

		sys_state.append([pred_tp, k+1, player_time, playback_time, server_time, buffer_length, state, last_bit_rate, current_reward, seq])

	# Filter sys_state to k'
	if pruning:
		reward_list = []
		for s in range(len(sys_state)):
			reward_list += [[sys_state[s][-2], sys_state[s][-3], s]]

		reward_list.sort(reverse=True)
		new_sys_state = []
		for i in range(K_P):
			new_sys_state += [sys_state[reward_list[i][2]]]
		sys_state = new_sys_state

	k += 1
	if k == MPC_STEP:
		# print len(sys_state)
		return sys_state
	else:
		# print sys_state
		current_len = len(sys_state)
		j = 0
		while j < current_len:
			# print sys_state, k, j
			# print "<?????>"
			sys_state.extend(mpc_solver_seg(sys_state[0], pruning))
			# print "<?------?>"
			sys_state.pop(0)
			# print sys_state
			# print len(sys_state)
			j += 1
	return sys_state

def mpc_find_opt_seg(all_states):
	opt_reward = float("-inf")
	opt_action = -1
	for state in all_states:
		if state[8] > opt_reward:
			opt_action = state[9]
			opt_reward = state[8]
	return opt_action, opt_reward

def mpc_find_action_seg(mpc_input, pruning=False):
	all_states = mpc_solver_seg(mpc_input, pruning)
	return mpc_find_opt_seg(all_states)

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

