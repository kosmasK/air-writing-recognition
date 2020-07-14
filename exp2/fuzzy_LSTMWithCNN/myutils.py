import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import datasets, transforms
import csv
import random 
import sys
sys.path.append("../")
from tools import coorToImg, do_the_scaling, checkforDuplicates
from skimage.transform import rescale


# class CNNFrameLevelDataset(Dataset):
class FrameLevelDataset(Dataset):
	def __init__(self, static_feats, trail_feats, labels, trail_featrep):
		"""
			feats: Python list of numpy arrays that contain the sequence features.
				   Each element of this list is a numpy array of shape seq_length x feature_dimension
			labels: Python list that contains the label for each sequence (each label must be an integer)
		"""
		self.labels = labels

		# Static
		self.static_feats = static_feats

		# Trail
		self.lengths = [sample.shape[1] for sample in trail_feats] # Find the lengths 
		if trail_featrep=='rescale':
			self.trail_feats = self.rescale(trail_feats)
		elif trail_featrep=='zeropad':
			self.trail_feats = self.zero_pad_and_stack(trail_feats)
		elif trail_featrep=='identical':
			self.trail_feats = trail_feats

	def rescale(self, x):
		maxLen = max(self.lengths)
		x = [ rescale(sample, (1, maxLen / sample.shape[1])) for sample in x ]
		return x

	def zero_pad_and_stack(self, x):
		maxLen = max(self.lengths)
		padded = np.zeros((len(x), 2, maxLen))
		for i, sample in enumerate(x):
			sequence_length = sample.shape[1]
			padded[i,:,:sequence_length] = sample 
			
		return padded

	def __getitem__(self, item):
		return self.static_feats[item], self.trail_feats[item], self.labels[item], self.lengths[item]

	def __len__(self):
		return len(self.trail_feats)



def data_generator(args, root, featrep): 

	# Gather All Recordings' Filenames per Participant
	FDict = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[], 'J':[]}
	listdir = os.listdir(root)
	random.seed(args.seed)
	shuffled_listdir = random.sample(listdir, len(listdir))
	for filename in shuffled_listdir:
		FDict[filename.split('_')[0]].append(filename)

	# Count Valid Classes per Participant and Create Equal Class Dataset
	strToInt = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}	
	MinimalFDict = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[], 'J':[]}
	for k in FDict:
		Occ=[0]*10
		for filename in FDict[k]:
			i = strToInt[filename.split('_')[1]]
			Occ[i] += 1
			if Occ[i]>10: continue
			MinimalFDict[k].append(filename)
		# print(Occ)
	# print()

	# Count Valid Classes per Participant and Create Equal Class Dataset
	strToInt = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}
	for k in MinimalFDict:
		Occ=[0]*10
		for filename in MinimalFDict[k]:
			i = strToInt[filename.split('_')[1]]
			Occ[i] += 1
		# print(Occ)


	# Separate input from groundtruth
	X = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[], 'J':[]}
	y = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[], 'J':[]}
	FNames = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[], 'J':[]}
	for k in MinimalFDict:
		for filename in MinimalFDict[k]:
			npzfile = np.load(root+'/'+filename)
			feats = npzfile['input']
			if featrep=='image':
				feats = coorToImg(feats)	
			if featrep=='static':
				feats = coorToStaticFeats(feats)

			X[k].append(feats)
			y[k].append(npzfile['target'])
			FNames[k].append(filename)

	# Convert data to torch to tensors 
	for k in X:
		for data in [X[k], y[k]]:
			for i in range(len(data)):
				data[i] = torch.Tensor(data[i].astype(np.float64))

	# List form
	# ignore='D'
	XFold=[]
	YFold=[]
	FFold=[]
	for k in MinimalFDict:
		# if k==ignore: continue
		XFold.append(X[k])
		YFold.append(y[k])
		FFold.append(FNames[k])

	# # Check Algorithm Correctness (check for file duplicates)
	# if checkforDuplicates(FNames):
	# 	print('Duplicates appear in the dataset!!' )
	# 	exit(1)

	# Check Algorithm Correctness (disjoint folds)
	for i in range(args.n_folds):
		for j in range(args.n_folds):
			if i==j: continue
			isdisjoint = set(FFold[i]).isdisjoint(set(FFold[j]))
			if not isdisjoint:
				print('There is at least one couple of folds which are non-disjoint!!')
				exit(1)

	return XFold, YFold, FFold



def aux_fold_generator(args, XFold, YFold, FFold):

	N=args.n_folds
	for fold_id in range(args.n_folds):

		X_test = XFold[fold_id] 	# e.g. fold_id = 0
		y_test = YFold[fold_id]
		F_test = FFold[fold_id]

		# print('test_id:', fold_id, end = ' ')
		X_valid = XFold[(fold_id+1)%N]	# e.g. fold_id%N+1 = 1
		y_valid = YFold[(fold_id+1)%N]
		F_valid = FFold[(fold_id+1)%N]
		# print('valid_id:', (fold_id+1)%N, end = ' ')

		X_train=[]
		y_train=[]
		F_train=[]
		# print('train_ids:', end = ' ')
		for i in range(N-2):
			X_train+=XFold[(fold_id+i+2)%N]
			y_train+=YFold[(fold_id+i+2)%N]
			F_train+=FFold[(fold_id+i+2)%N]
			# print((fold_id+i+2)%N, end = ' ')
		# print()

		# ################ Ensure Equal Class Distribution on Validation and Train Sets ###################

		# if args.particip_cross:

		X = X_train + X_valid
		y = y_train + y_valid
		F = F_train + F_valid
		idx_list = np.arange(len(y), dtype="int32")
		np.random.seed(args.seed)
		np.random.shuffle(idx_list)

		# random.seed(args.seed)
		# y = random.sample(y, len(y))
		Occ=[0]*10
		ValIdx=[]
		for i, t in enumerate(y):
			t = int(t)
			Occ[t] += 1
			if Occ[t]>10: continue
			ValIdx.append(i)

		X_valid = [X[i] for i in idx_list if i in ValIdx]
		X_train = [X[i] for i in idx_list if i not in ValIdx]
		y_valid = [y[i] for i in idx_list if i in ValIdx]
		y_train = [y[i] for i in idx_list if i not in ValIdx]
		F_valid = [F[i] for i in idx_list if i in ValIdx]
		F_train = [F[i] for i in idx_list if i not in ValIdx]

		# ################################################################################################

		### NOTE ###
		# Check Algorithm Correctness (check for file duplicates)
		if checkforDuplicates(F_train + F_valid + F_test):
			print('Duplicates appear in the dataset!!' )
			exit(1)

		yield X_train, y_train, X_valid, y_valid, X_test, y_test, F_train, F_valid, F_test


def fold_generator(args, XFold_static, YFold_static, FFold_static, XFold_trail, YFold_trail, FFold_trail):
	# XFold_static, YFold_static, FFold_static = data_generator(args, root, featrep='image')
	# XFold_trail, YFold_trail, FFold_trail = data_generator(args, root, featrep='zeropad')

	for fold_id, (static_data, trail_data) in enumerate( zip(aux_fold_generator(args, XFold_static, YFold_static, FFold_static), aux_fold_generator(args, XFold_trail, YFold_trail, FFold_trail))):

		X_static_train, y_static_train, X_static_valid, y_static_valid, X_static_test, y_static_test, \
		F_static_train, F_static_valid, F_static_test = static_data

		X_trail_train, y_trail_train, X_trail_valid, y_trail_valid, X_trail_test, y_trail_test, \
		F_trail_train, F_trail_valid, F_trail_tetst = trail_data

		# Just Checking # 
		if not y_static_train==y_trail_train:
			print('Error when shuffling!')
			exit(1)

		# Just Checking # 
		if not y_static_valid==y_trail_valid:
			print('Error when shuffling!')
			exit(1)

		# Just Checking # 
		if not y_static_test==y_trail_test:
			print('Error when shuffling!')
			exit(1)

		# Just Checking 
		if not F_trail_train==F_static_train:
			print('Error when shuffling!')
			exit(1)

		y_train = y_static_train
		y_valid = y_static_valid
		y_test = y_static_test

		train_set = FrameLevelDataset(X_static_train, X_trail_train, y_train, trail_featrep=args.featrep)
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

		val_set = FrameLevelDataset(X_static_valid, X_trail_valid, y_valid, trail_featrep=args.featrep)
		val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

		test_set = FrameLevelDataset(X_static_test, X_trail_test, y_test, trail_featrep=args.featrep)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

		yield train_loader, val_loader, test_loader, F_static_test