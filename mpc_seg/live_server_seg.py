import numpy as np

SEG_DURATION = 1000.0
SERVER_START_UP_TH = 2000.0				# <========= TO BE MODIFIED. TEST WITH DIFFERENT VALUES
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
	def __init__(self, seg_duration, start_up_th, randomSeed = RANDOM_SEED):
		np.random.seed(randomSeed)
		self.seg_duration = seg_duration
		self.time = start_up_th + np.random.randint(1,seg_duration)
		self.start_up_th = start_up_th
		self.current_seg_idx = -1
		self.segs = []	
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		self.init_encoding()
	

	def init_encoding(self):
		# assert self.ratio
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
		self.next_delivery = self.segs.pop(0)
		
	def encoding_update(self, starting_time, end_time):
		temp_time = starting_time
		while True:
			next_time = (int(temp_time/self.seg_duration) + 1) * self.seg_duration
			if next_time > end_time:
				break
			# Generate segs and insert to encoding buffer
			temp_time = next_time
			self.current_seg_idx += 1
			self.generate_seg_size()
				# print self.current_seg_size
			self.segs.append([self.current_seg_idx, \
								self.current_seg_size])	# for 1s segment
			
	def update(self, downloadig_time):
		pre_time = self.time
		self.time += downloadig_time
		self.encoding_update(pre_time, self.time)
		return self.time

	def sync_encoding_buffer(self):
		target_encoding_len = 0
		new_heading_time = 0.0
		missing_count = 0
		target_encoding_len = self.start_up_th/self.seg_duration
		while not len(self.segs) == target_encoding_len:
			self.segs.pop(0)
			missing_count += 1
		new_heading_time = self.segs[0][0] * self.seg_duration
		return new_heading_time, missing_count

	def generate_seg_size(self):
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		encoding_coef = 1.0
		estimate_seg_size = [x * encoding_coef for x in BITRATE]
		self.current_seg_size = estimate_seg_size 


	def wait(self):
		next_available_time = (int(self.time/self.seg_duration) + 1) * self.seg_duration
		self.encoding_update(self.time, next_available_time)
		assert len(self.segs) == 1
		time_interval = next_available_time - self.time
		self.time = next_available_time
		return time_interval 

	def test_reset(self, start_up_th):
		self.time = start_up_th + np.random.randint(1,self.seg_duration)		# start from 2000ms	
		self.start_up_th = start_up_th
		self.current_seg_idx = -1
		self.segs = []	
		self.current_seg_size = [[] for i in range(len(BITRATE))]
		self.encoding_update(0.0, self.time)
		self.next_delivery = []
		# self.delay_tol = start_up_th
	def check_segs_empty(self):
		if len(self.segs) == 0:
			return True
		else: return False

def main():
	server = Live_Server(seg_duration=SEG_DURATION, start_up_th=SERVER_START_UP_TH)
	server.init_encoding()
	print(server.segs, server.time)
	print(server.next_delivery)


if __name__ == '__main__':
	main()