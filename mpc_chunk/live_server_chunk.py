import numpy as np

SEG_DURATION = 1000.0
# FRAG_DURATION = 1000.0
CHUNK_DURATION = 200.0
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES

# CHUNK_IN_SEG = int(SEG_DURATION/CHUNK_DURATION)		# 4
# CHUNK_IN_FRAG = int(FRAG_DURATION/FRAG_DURATION)	# 2
# FRAG_IN_SEG = int(SEG_DURATION/FRAG_DURATION)		# 2 or 5
# ADD_DELAY = 3000.0
RANDOM_SEED = 10
MS_IN_S = 1000.0
KB_IN_MB = 1000.0
# New bitrate setting, 6 actions, correspongding to 240p, 360p, 480p, 720p, 1080p and 1440p(2k)
BITRATE = [300.0, 500.0, 1000.0, 2000.0, 3000.0, 6000.0]
# BITRATE = [300.0, 6000.0]
# BITRATE = [500.0, 2000.0, 5000.0, 8000.0, 16000.0]	# 5 actions

BITRATE_LOW_NOISE = 0.7
BITRATE_HIGH_NOISE = 1.3
RATIO_LOW_2 = 2.0				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_2 = 10.0			# This is the highest ratio between first chunk and the sum of all others
RATIO_LOW_5 = 0.75				# This is the lowest ratio between first chunk and the sum of all others
RATIO_HIGH_5 = 1.0			# This is the highest ratio between first chunk and the sum of all others
EST_LOW_NOISE = 0.98
EST_HIGH_NOISE = 1.02


class Live_Server(object):
	def __init__(self, seg_duration, chunk_duration, start_up_th, randomSeed = RANDOM_SEED):
		np.random.seed(randomSeed)
		self.seg_duration = seg_duration
		self.chunk_duration = chunk_duration
		self.chunk_in_seg = seg_duration/chunk_duration

		self.time = start_up_th + np.random.randint(1,seg_duration)
		self.start_up_th = start_up_th
		self.current_seg_idx = -1
		self.current_chunk_idx = 0
		self.chunks = []	# 1 for initial chunk, 0 for following chunks
		self.ratio = None
		self.chunks = []	# 1 for initial chunk, 0 for following chunks
		self.current_seg_size = [[] for i in range(len(BITRATE))]

	def set_ratio(self, ratio):
		self.ratio = ratio

	def init_encoding(self):
		assert self.ratio
		self.encoding_update(0.0, self.time)
		self.next_delivery = []
		self.generate_next_delivery()

	def get_next_delivery(self):
		return self.next_delivery

	def clean_next_delivery(self):
		self.next_delivery = []

	def get_time(self):
		return self.time

	def generate_next_delivery(self):
		deliver_chunks = []
		deliver_chunks.append(self.chunks.pop(0))
		#deliver_end = 0
		#for i in range(len(self.chunks)):
		#	if not self.chunks[i][0] == deliver_chunks[-1][0]:
		#		break
		#	deliver_end += 1
		#deliver_chunks.extend(self.chunks[:deliver_end])
		#del self.chunks[:deliver_end]
		self.next_delivery.extend(deliver_chunks[0][:2])
		self.next_delivery.append(deliver_chunks[-1][1])
		delivery_sizes = []
		for i in range(len(BITRATE)):
			delivery_sizes.append(np.sum([chunk[2][i] for chunk in deliver_chunks]))
		self.next_delivery.append(delivery_sizes)
		
	def encoding_update(self, starting_time, end_time):
		temp_time = starting_time
		while True:
			next_time = (int(temp_time/self.chunk_duration) + 1) * self.chunk_duration
			if next_time > end_time:
				break
			# Generate chunks and insert to encoding buffer
			temp_time = next_time
			if next_time%self.seg_duration == self.chunk_duration:
			# If it is the first chunk in a seg
				self.current_seg_idx += 1
				self.current_chunk_idx = 0
				self.generate_chunk_size()
				# print self.current_seg_size
				self.chunks.append([self.current_seg_idx, self.current_chunk_idx, \
									[chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size],\
									[np.sum(chunk_size) for chunk_size in self.current_seg_size]])	# for 2s segment
			else:
				self.current_chunk_idx += 1
				# print(self.current_chunk_idx, self.current_seg_size)
				self.chunks.append([self.current_seg_idx, self.current_chunk_idx, [chunk_size[self.current_chunk_idx] for chunk_size in self.current_seg_size]])

	def update(self, downloadig_time):
		pre_time = self.time
		self.time += downloadig_time
		self.encoding_update(pre_time, self.time)
		return self.time

	def sync_encoding_buffer(self):
		target_encoding_len = 0
		new_heading_time = 0.0
		missing_count = 0
		# Modified for both 200 and 500 ms
		num_chunks = int((self.time%self.seg_duration)/self.chunk_duration)
		target_encoding_len = self.start_up_th/self.chunk_duration + num_chunks
		# Old, for 500
		# if self.time%self.frag_duration >= CHUNK_DURATION:
		# 	target_encoding_len = self.start_up_th/CHUNK_DURATION + 1
		# else:
		# 	target_encoding_len = self.start_up_th/CHUNK_DURATION
		# print(len(self.chunks))
		while not len(self.chunks) == target_encoding_len:
			self.chunks.pop(0)
			missing_count += 1
		new_heading_time = self.chunks[0][0] * self.seg_duration + self.chunks[0][1] * self.chunk_duration
		assert self.chunks[0][1] == 0
		# self.generate_next_delivery()
		return new_heading_time, missing_count

	# chunk size for next/current segment
	def generate_chunk_size(self):
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		encoding_coef = 1.0
		estimate_seg_size = [x * encoding_coef for x in BITRATE]

		if self.chunk_in_seg == 2:
		# Distribute size for chunks, currently, it should depend on chunk duration (200 or 500)
			# seg_ratio = [np.random.uniform(EST_LOW_NOISE*ratio, EST_HIGH_NOISE*ratio) for x in range(len(BITRATE))]
			for i in range(len(estimate_seg_size)):
				temp_aux_chunk_size = estimate_seg_size[i]/(1+self.ratio)
				temp_ini_chunk_size = estimate_seg_size[i] - temp_aux_chunk_size
				self.current_seg_size[i].extend((temp_ini_chunk_size, temp_aux_chunk_size))
		# if 200ms, needs to be modified, not working
		elif self.chunk_in_seg == 5:
			for i in range(len(estimate_seg_size)):
				temp_ini_chunk_size = estimate_seg_size[i] * self.ratio / (1 + self.ratio)
				temp_aux_chunk_size = (estimate_seg_size[i] - temp_ini_chunk_size) / (self.chunk_in_seg - 1)
				temp_chunks_size = [temp_ini_chunk_size]
				temp_chunks_size.extend([temp_aux_chunk_size for _ in range(int(self.chunk_in_seg) - 1)])
				self.current_seg_size[i].extend(temp_chunks_size)

	def wait(self):
		next_available_time = (int(self.time/self.chunk_duration) + 1) * self.chunk_duration
		self.encoding_update(self.time, next_available_time)
		assert len(self.chunks) == 1
		time_interval = next_available_time - self.time
		self.time = next_available_time
		return time_interval 

	def test_reset(self, start_up_th):
		self.time = start_up_th + np.random.randint(1,self.seg_duration)		# start from 2000ms	
		self.start_up_th = start_up_th
		self.current_seg_idx = -1
		self.current_chunk_idx = 0
		self.chunks = []	# 1 for initial chunk, 0 for following chunks
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		self.encoding_update(0.0, self.time)
		self.next_delivery = []
		# self.delay_tol = start_up_th
	def check_chunks_empty(self):
		if len(self.chunks) == 0:
			return True
		else: return False

def main():
	server = Live_Server(seg_duration=SEG_DURATION, chunk_duration=CHUNK_DURATION, start_up_th=SERVER_START_UP_TH)
	server.set_ratio(0.8)
	server.init_encoding()
	print(server.chunks, server.time)
	print(server.next_delivery)


if __name__ == '__main__':
	main()
