'''
Created on Feb 2020

@author: grigoris
'''

import json 
import os
import numpy as np
import shutil
from functools import reduce
import argparse
import sys
sys.path.append("../")
from exp1.tools import str2bool

def data_for_scaling(path_to_data, subdir):

	for filename in os.listdir(path_to_data+subdir):
		if '_calibration' in filename:
			calib_file = filename

	with open(path_to_data+subdir+'/'+calib_file) as json_file:
		data = json.load(json_file)
		lista = data['vertices']
		# print(lista)

		leftX = min(lista[0]['x'], lista[3]['x'])
		rightX = max(lista[1]['x'], lista[2]['x'])
		lowerY = min(lista[2]['y'], lista[3]['y'])
		upperY = max(lista[0]['y'], lista[1]['y'])

		minim = [leftX, lowerY]
		maxim = [rightX, upperY]
		# print(minim)
		# print(maxim)
	return np.array(minim), np.array(maxim)

def do_the_scaling(vector, minim, maxim):
	numerator = vector - minim
	denominator = maxim - minim
	return numerator/denominator


def create_list_of_faulty_recordings(path_to_data):
	NotRight=[]
	for subdir in os.listdir(path_to_data):
		print(subdir)
		if os.path.isfile(path_to_data+subdir): # parse "not_right' file here
			with open(path_to_data+subdir) as not_right:
				for line in not_right:
					alist = line.split('/')[-1].split('_')[0:3]
					keyword = '_'.join(alist)
					NotRight.append(keyword+'_trail.json')
	return NotRight

if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Data Preparation')
	parser.add_argument('--new_shuffling', type=str2bool, default=False, metavar='N',
						help='batch size (default: 64)')

	args = parser.parse_args()

	n_test = 200
	strToInt = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 
				'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}

	path_to_data = "../Recordings/"

	# Create path_ot_store
	path_to_store = "../data/air_trails/"
	try:
		shutil.rmtree(path_to_store)
	except:
		print('Error while deleting directory')
	os.mkdir(path_to_store)
	os.mkdir(path_to_store+'Train/')
	os.mkdir(path_to_store+'Test/')
	
	NotRight = create_list_of_faulty_recordings(path_to_data)

	count=0
	countNotRight=0
	Seqs=[]
	Targets=[]
	Filenames=[]
	for subdir in os.listdir(path_to_data):
		# print(subdir)
		if os.path.isfile(path_to_data+subdir): # don't parse "not_right' file here
			continue

		minim, maxim = data_for_scaling(path_to_data, subdir)
		for filename in os.listdir(path_to_data+subdir):

			# Filter out some files
			if filename in NotRight:
				countNotRight+=1
				continue

			# Parse trail files and normalize xy coordinates
			if '_trail' in filename:
				seq=[]
				number = filename.split("_")[1]
				filenametostore = '_'.join(filename.split("_")[:-1])		
				target = strToInt[number]
				with open(path_to_data+subdir+'/'+filename) as json_file:
					data = json.load(json_file)
					for lista in data["data"]:
						id = lista[0]
						xyz = lista[1]
						xy = xyz[:-1]
						z = xyz[-1]
						xy = do_the_scaling(np.array(xy), minim, maxim)
						seq.append(xy)

				seq = np.array(seq).T
				Filenames.append(filenametostore)
				Targets.append(target)
				Seqs.append(seq)
				count+=1


	print("count", count)

	if args.new_shuffling:
		# Shuffle ids and create Train and Test datasets
		shuffle = np.random.permutation(count)
		TestX, TestY, TrainX, TrainY = [], [], [], []
		TestIds, TrainIds, TestFiles, TrainFiles = [], [], [], []
		for i in shuffle:
			# print(np.sum( TestY.count(Targets[i]) ))
			if np.sum( TestY.count(Targets[i]) ) < 20:
				TestIds += [i]
				TestX += [Seqs[i]]
				TestY += [Targets[i]]
				TestFiles += [Filenames[i]]
			else:
				TrainIds += [i]
				TrainX += [Seqs[i]]
				TrainY += [Targets[i]]
				TrainFiles += [Filenames[i]]

		print(TestIds)

	else:

		TestIds = [776, 490, 678, 315, 77, 162, 485, 320, 1084, 974, 698, 32, 378, 732, 859, 825, 127, 827, 424, 110, 1136, 656, 466, 450, 269, 157, 1076, 514, 298, 277, 541, 261, 
				188, 683, 373, 254, 473, 397, 30, 580, 1056, 234, 474, 716, 633, 689, 1002, 1055, 180, 912, 226, 1062, 212, 488, 372, 592, 1018, 773, 836, 951, 348, 459, 1023, 881, 443, 508, 1073, 94, 1044, 915, 889, 803, 192, 1104, 890, 929, 73, 1003, 547, 925, 209, 907, 996, 201, 481, 268, 991, 171, 248, 981, 878, 831, 768, 335, 777, 15, 400, 703, 735, 1053, 486, 48, 941, 1045, 842, 695, 516, 922, 667, 352, 1019, 498, 810, 84, 407, 783, 1020, 694, 674, 613, 66, 848, 801, 1032, 158, 455, 731, 728, 229, 454, 931, 1004, 562, 46, 203, 328, 343, 36, 796, 759, 738, 606, 309, 559, 
				232, 909, 517, 711, 109, 557, 163, 887, 165, 531, 856, 910, 452, 857, 808, 686, 193, 176, 3, 550, 356, 257, 871, 502, 809, 5, 365, 63, 160, 462, 552, 702, 231, 381, 484, 643, 103, 284, 387, 1089, 935, 526, 551, 57, 747, 1111, 849, 658, 412, 161, 421, 965, 528, 1081, 539, 61]

		TrainIds = [i for i in range(count) if not i in TestIds]

		TestX  = [Seqs[i] for i in TestIds]
		TrainX = [Seqs[i] for i in TrainIds]
		TestY  = [Targets[i] for i in TestIds]
		TrainY = [Targets[i] for i in TrainIds]
		TestFiles  = [Filenames[i] for i in TestIds]
		TrainFiles = [Filenames[i] for i in TrainIds]

	print()
	print("Problem if common files are stored in this list", list(set(TrainFiles).intersection(TestFiles)) )
	print()

	# Save Test
	for i in range(len(TestIds)):
		np.savez(path_to_store+'Test/'+TestFiles[i]+'.npz', input=TestX[i], target=TestY[i])

	# Save Train
	for i in range(len(TrainIds)):
		np.savez(path_to_store+'Train/'+TrainFiles[i]+'.npz', input=TrainX[i], target=TrainY[i])


	print(countNotRight, 'files were neglected out of', count+countNotRight)
	print('Train samples:', len(TrainIds))
	print('Test samples:', len(TestIds))

