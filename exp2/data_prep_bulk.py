'''
Created on June 2020

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
from exps.tools import str2bool

def data_for_scaling(path_to_data, subdir):

	for filename in os.listdir(path_to_data+subdir):
		if '_calibration' in filename:
			calib_file = filename

	with open(path_to_data+subdir+'/'+calib_file) as json_file:
		data = json.load(json_file)
		lista = data['vertices']

		leftX = min(lista[0]['x'], lista[3]['x'])
		rightX = max(lista[1]['x'], lista[2]['x'])
		lowerY = min(lista[2]['y'], lista[3]['y'])
		upperY = max(lista[0]['y'], lista[1]['y'])

		minim = [leftX, lowerY]
		maxim = [rightX, upperY]

	return np.array(minim), np.array(maxim)

def do_the_scaling(vector, minim, maxim):
	numerator = vector - minim
	denominator = maxim - minim
	return numerator/denominator


def create_list_of_faulty_recordings(path_to_data):
	NotRight=[]
	for subdir in os.listdir(path_to_data):
		# print(subdir)
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
	parser.add_argument('--keep_left', type=str2bool, default=False)

	args = parser.parse_args()

	strToInt = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 
				'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}

	path_to_data = "../Recordings/"

	# Create path_ot_store
	path_to_store = "../data/air_bulk_trails/"
	if args.keep_left:
		path_to_store = "../data/air_bulk_trails_all/"

	try:
		shutil.rmtree(path_to_store)
	except:
		print('Error while deleting directory (not important)')
	os.mkdir(path_to_store)

	
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
				if not args.keep_left:
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

				# Data to store
				Filenames.append(filenametostore)
				Targets.append(target)
				Seqs.append(seq)

				count+=1

	print(count, 'files will be used.')
	# print(countNotRight, 'files were neglected out of', count+countNotRight, '.')
	print(countNotRight, 'files were neglected.')

	# Save Test
	for i in range(len(Filenames)):
		np.savez(path_to_store+Filenames[i]+'.npz', input=Seqs[i], target=Targets[i])

