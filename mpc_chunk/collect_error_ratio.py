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
		mpc_ori_value = {}
		with open(ERROR_DIR+'MPC_'+lat+'.txt') as opt_f:
			for line in opt_f:
				parse = line.strip('\n')
				parse = line.split()
				mpc_ori_value[parse[0]] = float(parse[1])

		for file in os.listdir(ERROR_DIR):
			file_path = ERROR_DIR + file
			print(file)
			if 'txt' in file and lat in LATENCY and not 'MPC' in file: 
				with open(file_path, 'rb') as f:
					value = []
					for line in f:
						parse = line.strip('\n')
						parse = line.split()
						value.append(max(parse[0], 0))
					value.append(mpc_ori_value[file])
				print(value)

if __name__ == '__main__':
	main()