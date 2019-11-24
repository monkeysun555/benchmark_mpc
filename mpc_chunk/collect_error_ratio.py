import os
import numpy as np
import mpc_solver_chunk as mpc
import math

ERROR_DIR = './error/'
OPTIMAL_PATH = '.'
LATENCY = ['2s', '3s', '4s']

def main():
	for lat in LATENCY:
		ratios = []
		# mpc_ori_value = {}
		# with open(ERROR_DIR+'MPC_'+lat+'.txt') as opt_f:
		# 	for line in opt_f:
		# 		parse = line.strip('\n')
		# 		parse = line.split()
		# 		mpc_ori_value[parse[0]] = float(parse[1])

		for file in os.listdir(ERROR_DIR):
			file_path = ERROR_DIR + file
			if 'txt' in file and lat in file: 
				print(file)	
				with open(file_path, 'rb') as f:
					value = []
					for line in f:
						parse = line.strip('\n')
						parse = line.split()
						value.append(max(float(parse[1]), 0))
				# print(value)
				ratios.append(value)
		# print(ratios)
		## Get all values under on buffer length
		# Calculate ratios
		for ratio in ratios:
			opt = ratio[0]
			for r_idx in range(len(ratio)):
				ratio[r_idx] = ratio[r_idx]/opt
		ave_ratios = []
		for r_idx in range(len(ratios[0])):
				ave_ratios.append(np.mean( [ratios[rs_idx][r_idx]  for rs_idx in range(len(ratios))]))
		print(ave_ratios)
if __name__ == '__main__':
	main()