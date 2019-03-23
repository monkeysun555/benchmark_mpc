import numpy as np

RANDOM_SEED = 13
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]	# 5 actions
PACKET_PAYLOAD_PORTION = 0.973	# 1460/1500

RTT_LOW = 30.0
RTT_HIGH = 40.0 
SEG_RANDOM_RATIO_LOW = 0.95
SEG_RANDOM_RATIO_HIGH = 1.05

MS_IN_S = 1000.0	# in ms
KB_IN_MB = 1000.0	# in ms

class Live_Player(object):
	def __init__(self, time_traces, throughput_traces, seg_duration, start_up_th, freezing_tol, latency_tol, randomSeed = RANDOM_SEED):
		np.random.seed(randomSeed)

		self.time_traces = time_traces
		self.throughput_traces = throughput_traces

		self.trace_idx = np.random.randint(len(self.throughput_traces))
		self.throughput_trace = self.throughput_traces[self.trace_idx]
		self.time_trace = self.time_traces[self.trace_idx]

		self.playing_time = 0.0
		self.time_idx = np.random.randint(1,len(self.time_trace))
		self.last_trace_time = self.time_trace[self.time_idx-1] * MS_IN_S	# in ms

		self.seg_duration = seg_duration
		self.buffer = 0.0	# ms
		self.state = 0	# 0: start up.  1: traceing. 2: rebuffering
		self.start_up_th = start_up_th
		self.freezing_tol = freezing_tol
		self.latency_tol = latency_tol
		self.bw = [0.5]*BW_HIS_LEN

		print('player initial finish')

	def fetch(self, quality, next_seg_set, seg_idx, playing_speed = 1.0):
		# Action initialization
		start_state = self.state
		seg_size = next_seg_set[quality] # in Kbits not KBytes
		seg_start_time = seg_idx * self.seg_duration
		# as mpd is based on prediction, there is noise
		seg_size = np.random.uniform(SEG_RANDOM_RATIO_LOW*seg_size, SEG_RANDOM_RATIO_HIGH*seg_size)
		seg_sent = 0.0	# in Kbits
		downloading_fraction = 0.0	# in ms
		freezing_fraction = 0.0	# in ms
		time_out = 0
		rtt = 0.0
		# Handle RTT 
		# Tick here, for seg mode, always has RTT
		take_action = 1
		if take_action:
			rtt = np.random.uniform(RTT_LOW, RTT_HIGH) 	# in ms
			duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time	# in ms
			if duration > rtt:
				self.last_trace_time += rtt
			else:
				temp_rtt = rtt - duration
				self.last_trace_time += duration
				self.time_idx += 1
				if self.time_idx >= len(self.time_trace):
					self.time_idx = 1
					self.last_trace_time = 0.0
				self.last_trace_time += temp_rtt
				assert self.last_trace_time < self.time_trace[self.time_idx] * MS_IN_S
			downloading_fraction += rtt
			assert self.state == 1 or self.state == 0
			# Check whether during startup
			if self.state == 1:
				self.playing_time += np.minimum(self.buffer, playing_speed*rtt)			# modified based on playing speed, adjusted, * speed
				freezing_fraction += np.maximum(rtt - self.buffer/playing_speed, 0.0)	# modified based on playing speed, real time, /speed
				self.buffer = np.maximum(0.0, self.buffer - playing_speed*rtt)			# modified based on playing speed, adjusted, * speed
				# chech whether enter freezing
				if freezing_fraction > 0.0:
					self.state = 2
			else:
				freezing_fraction += rtt 	# in ms
		# Seg downloading
		while True:
			throughput = self.throughput_trace[self.time_idx]	# in Mbps or Kbpms
			duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time		# in ms
			deliverable_size = throughput * duration * PACKET_PAYLOAD_PORTION	# in Kbits		
			# Will also check whether freezing time exceeds the TOL
			if deliverable_size + seg_sent > seg_size:
				fraction = (seg_size - seg_sent) / (throughput * PACKET_PAYLOAD_PORTION)	# in ms, real time
				if self.state == 1:
					assert freezing_fraction == 0.0
					temp_freezing = np.maximum(fraction - self.buffer/playing_speed, 0.0)		# modified based on playing speed
					if temp_freezing > self.latency_tol:
						# should not happen
						time_out = 1
						self.last_trace_time += self.buffer + self.freezing_tol
						downloading_fraction += self.buffer + self.freezing_tol
						self.playing_time += self.buffer
						seg_sent += (self.freezing_tol + self.buffer) * throughput * PACKET_PAYLOAD_PORTION	# in Kbits	
						self.state = 0
						self.buffer = 0.0
						assert seg_sent < seg_size
						return seg_sent, downloading_fraction, freezing_fraction, time_out, start_state

					downloading_fraction += fraction
					self.last_trace_time += fraction
					freezing_fraction += np.maximum(fraction - self.buffer/playing_speed, 0.0)	# modified based on playing speed 
					self.playing_time += np.minimum(self.buffer, playing_speed*fraction)		# modified based on playing speed 
					self.buffer = np.maximum(self.buffer - playing_speed*fraction, 0.0)			# modified based on playing speed 
					if np.round(self.playing_time + self.buffer, 2) == np.round(seg_start_time, 2):
						self.buffer += self.seg_duration
					else:
						# Should not happen in general case, this is constrain for training
						self.buffer = self.seg_duration
						self.playing_time = seg_start_time
					break
				# Freezing
				elif self.state == 2:
					assert self.buffer == 0.0
					if freezing_fraction + fraction > self.freezing_tol:
						time_out = 1
						self.last_trace_time += self.freezing_tol - freezing_fraction
						downloading_fraction += self.freezing_tol - freezing_fraction
						seg_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION	# in Kbits
						freezing_fraction = self.freezing_tol
						self.state = 0
						assert seg_sent < seg_size
						return seg_sent, downloading_fraction, freezing_fraction, time_out, start_state
					freezing_fraction += fraction
					self.last_trace_time += fraction
					downloading_fraction += fraction
					self.buffer += self.seg_duration
					self.playing_time = seg_start_time
					self.state = 1
					break

				else:
					assert self.buffer < self.start_up_th
				
					downloading_fraction += fraction
					self.buffer += self.seg_duration
					freezing_fraction += fraction
					self.last_trace_time += fraction
					if self.buffer >= self.start_up_th:
						# Because it might happen after one long freezing (not exceed freezing tol)
						# And resync, enter initial phase
						buffer_end_time = seg_start_time + self.seg_duration
						self.playing_time = buffer_end_time - self.buffer
						# print(buffer_end_time, self.buffer)
						self.state = 1
					break

			# One seg downloading does not finish
			# traceing
			if self.state == 1:
				assert freezing_fraction == 0.0
				temp_freezing = np.maximum(duration - self.buffer/playing_speed, 0.0)		# modified based on playing speed
				self.playing_time += np.minimum(self.buffer, playing_speed*duration)		# modified based on playing speed
				# Freezing time exceeds tolerence
				if temp_freezing > self.freezing_tol:
					# should not happen
					time_out = 1
					self.last_trace_time += self.freezing_tol + self.buffer
					downloading_fraction += self.freezing_tol + self.buffer
					freezing_fraction = self.freezing_tol
					self.playing_time += self.buffer
					self.buffer = 0.0
					# exceed TOL, enter startup, freezing time equals TOL
					self.state = 0
					seg_sent += (self.freezing_tol + self.buffer) * throughput * PACKET_PAYLOAD_PORTION	# in Kbits
					assert seg_sent < seg_size
					return seg_sent, downloading_fraction, freezing_fraction, time_out, start_state

				seg_sent += duration * throughput * PACKET_PAYLOAD_PORTION	# in Kbits
				downloading_fraction += duration 	# in ms
				self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S	# in ms
				self.time_idx += 1
				if self.time_idx >= len(self.time_trace):
					self.time_idx = 1
					self.last_trace_time = 0.0	# in ms
				self.buffer = np.maximum(self.buffer - playing_speed*duration, 0.0)			# modified based on playing speed
				# update buffer and state
				if temp_freezing > 0:
					# enter freezing
					self.state = 2
					assert self.buffer == 0.0
					freezing_fraction += temp_freezing

			# Freezing during trace
			elif self.state == 2:
				assert self.buffer == 0.0
				if duration + freezing_fraction > self.freezing_tol:
					time_out = 1
					self.last_trace_time += self.freezing_tol - freezing_fraction	# in ms
					self.state = 0
					downloading_fraction += self.freezing_tol - freezing_fraction
					seg_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION	# in Kbits
					freezing_fraction = self.freezing_tol
					# Download is not finished, seg_size is not the entire seg
					assert seg_sent < seg_size
					return seg_sent, downloading_fraction, freezing_fraction, time_out, start_state

				freezing_fraction += duration 	# in ms
				seg_sent += duration * throughput * PACKET_PAYLOAD_PORTION	# in kbits
				downloading_fraction += duration 	# in ms
				self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S	# in ms
				self.time_idx += 1
				if self.time_idx >= len(self.time_trace):
					self.time_idx = 1
					self.last_trace_time = 0.0	# in ms
			# Startup
			else:
				assert self.buffer < self.start_up_th
				
				seg_sent += duration * throughput * PACKET_PAYLOAD_PORTION
				downloading_fraction += duration
				self.last_trace_time = self.time_trace[self.time_idx] * MS_IN_S	# in ms
				self.time_idx += 1
				if self.time_idx >= len(self.time_trace):
					self.time_idx = 1
					self.last_trace_time = 0.0	# in ms
				freezing_fraction += duration

		return seg_size, downloading_fraction, freezing_fraction, time_out, start_state

	def sync_playing(self, sync_time):
		self.buffer = 0
		self.state = 0
		self.playing_time = sync_time

	def get_buffer_len(self):
		return self.buffer

	def get_state(self):
		return self.state

	def get_time(self):
		return self.playing_time

	def get_time_idx(self):
		return self.time_idx

	def adjust_start_up_th(self, new_start_up_th):
		self.start_up_th = new_start_up_th
		return

	def update_bw(self, new_bw):
		self.bw = np.roll(self.bw, -1)
		self.bw[-1] = new_bw

	def wait(self, wait_time):
		# If live server does not have any available segs, need to wait
		assert self.buffer > wait_time
		self.buffer -= wait_time
		self.playing_time += wait_time
		past_wait_time = 0.0	# in ms
		while  True:
			duration = self.time_trace[self.time_idx] * MS_IN_S - self.last_trace_time
			if past_wait_time + duration > wait_time:
				self.last_trace_time += wait_time - past_wait_time
				break
			past_wait_time += duration
			self.last_trace_time += duration
			self.time_idx += 1
			if self.time_idx >= len(self.time_trace):
				self.time_idx = 1
				self.last_trace_time = 0.0
		return

	def check_resync(self, server_time):
		sync = 0
		if server_time - self.playing_time > self.latency_tol:
			sync = 1
		return sync

	def reset(self, start_up_th):
		self.playing_time = 0.0
		self.trace_idx = np.random.randint(len(self.throughput_traces))
		self.throughput_trace = self.throughput_traces[self.trace_idx]
		self.time_trace = self.time_traces[self.trace_idx]

		self.time_idx = np.random.randint(1,len(self.time_trace))
		self.last_trace_time = self.time_trace[self.time_idx-1]	* MS_IN_S # in ms
		# self.playing_time = self.time_trace[self.time_idx-1] # in ms
		# self.real_time = 0.0
		# self.real_frac = 0.0

		self.buffer = 0.0	# ms
		self.state = 0	# 0: start up.  1: traceing. 2: rebuffering
		self.start_up_th = start_up_th

	def test_reset(self, start_up_th, random_seed):
		np.random.seed(random_seed)
		self.playing_time = 0.0
		self.trace_idx += 1
		if self.trace_idx >= len(self.time_traces):
			self.trace_idx = 0
		self.throughput_trace = self.throughput_traces[self.trace_idx]
		self.time_trace = self.time_traces[self.trace_idx]

		self.time_idx = np.random.randint(1,len(self.time_trace))
		self.last_trace_time = self.time_trace[self.time_idx-1] * MS_IN_S	# in ms
		# self.playing_time = self.time_trace[self.time_idx-1] # in ms
		# self.real_time = 0.0
		# self.real_frac = 0.0

		self.buffer = 0.0	# ms
		self.state = 0	# 0: start up.  1: playing. 2: rebuffering
		self.start_up_th = start_up_th

