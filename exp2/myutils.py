import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import datasets, transforms
from skimage.transform import rescale
import sys
sys.path.append("../")
from tools import checkforDuplicates
import random


class AirDataset(Dataset):
	def __init__(self, x, labels, featrep):

		'''
		torch.utils.data.Dataset is an abstract class representing a dataset. 
		Your custom dataset should inherit Dataset and override the methods __len__ and __getitem__
		(https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
		'''

		self.lengths = [sample.shape[1] for sample in x] # Find the lengths 

		# self.x = self.zero_pad_and_stack(x)
		if featrep=='rescale':
			self.x = self.rescale(x)
		elif featrep=='zeropad':
			self.x = self.zero_pad_and_stack(x)
		elif featrep=='identical' or featrep=='image' or featrep=='static': # "image" is for cnn, "static" is for TCN_static
			self.x = x

		self.labels = labels

	def rescale(self, x):
		maxLen = max(self.lengths)
		# print(type(x), len(x), x[0].shape)
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
		return self.x[item], self.labels[item], self.lengths[item]

	def __len__(self):
		return len(self.x)

def do_the_scaling(vector, minim, maxim):
	numerator = vector - minim
	denominator = maxim - minim
	return numerator/denominator

def coorToImg(feats):
	w, h = 28, 28
	# NORMALIZE (AGAIN)
	minXY = np.amin(feats, axis=1)
	maxXY = np.amax(feats, axis=1)
	normXY = np.array([do_the_scaling(xy, minXY, maxXY) for xy in feats.T])

	# FILL IMG ARRAY and SPAGHETTI FEATS
	img = np.ones((w, h))
	normXY *= w-1
	normXY = normXY.astype(int)
	for xy in normXY:
		# Img
		row, col = xy[0], xy[1]
		img[row, col] = 0

	return img


def coorToStaticFeats(feats):
	w, h = 28, 28
	# NORMALIZE (AGAIN)
	minXY = np.amin(feats, axis=1)
	maxXY = np.amax(feats, axis=1)
	normXY = np.array([do_the_scaling(xy, minXY, maxXY) for xy in feats.T])

	# FILL IMG ARRAY and SPAGHETTI FEATS
	img = np.ones((w, h))
	feats = np.zeros(784)-0.4 # init to -0.4
	normXY *= w-1
	normXY = normXY.astype(int)
	for xy in normXY:
		# Img
		row, col = xy[0], xy[1]
		img[row, col] = 0
		# Convert to static spaghetti feats
		px = w*row + col
		left = 28*row + (col-1)
		right = 28*row + (col+1)
		lower = 28*(row-1) + (col+1)
		upper = 28*(row+1) + (col+1)		

		feats[px] = 0.9
		try:
			feats[left] = random.uniform(0.1, 0.6) if (feats[left] < 0.9) else feats[left]
		except IndexError:
			pass 
		try:	
			feats[right] = random.uniform(0.1, 0.6) if (feats[right] < 0.9) else feats[right]
		except IndexError:
			pass 
		try:
			feats[lower] = random.uniform(0.1, 0.6) if (feats[lower] < 0.9) else feats[lower]
		except IndexError:
			pass 
		try:	
			feats[upper] = random.uniform(0.1, 0.6) if (feats[upper] < 0.9) else feats[upper]
		except IndexError:
			pass 

	feats = np.array([feats])

	return feats


def data_generator(args, root):  # for participant corss-validation

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
			if args.featrep=='image':
				feats = coorToImg(feats)	
			if args.featrep=='static':
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



def data_generator_bulk(args, root): # for classic 10-fild cross validation
	# Load Train 
	X, y, FNames = [], [], []
	for filename in os.listdir(root):
	# for filename in MinimalFList:
		npzfile = np.load(root+'/'+filename)
		feats = npzfile['input']
		if args.featrep=='image':
			feats = coorToImg(feats)	
		if args.featrep=='static':
			feats = coorToStaticFeats(feats)

		X.append(feats)
		y.append(npzfile['target'])
		FNames.append(filename)

	# Shuffle Data
	idx_list = np.arange(len(X), dtype="int32")
	np.random.seed(args.seed)
	np.random.shuffle(idx_list)
	X = [X[i] for i in idx_list]
	y = [y[i] for i in idx_list]
	FNames = [FNames[i] for i in idx_list]

	# Check Algorithm Correctness (check for file duplicates)
	if checkforDuplicates(FNames):
		print('Duplicates appear in the dataset!!' )
		exit(1)

	# Convert data to torch to tensors 
	for data in [X, y]:
		for i in range(len(data)):
			data[i] = torch.Tensor(data[i].astype(np.float64))

	# Create Folds
	N=args.n_folds
	XFold=[]
	YFold=[]
	FFold=[]
	p = len(X)*(1/N)
	for i in range(N-1):
		print('Fold', i, 'indices:', round(i*p), round((i+1)*p))
		XFold.append(X[round(i*p):round((i+1)*p)])
		YFold.append(y[round(i*p):round((i+1)*p)])
		FFold.append(FNames[round(i*p):round((i+1)*p)])
	# last fold
	print('Fold', N-1, 'indices:', round((N-1)*p), len(X))
	XFold.append(X[round((N-1)*p):])
	YFold.append(y[round((N-1)*p):])
	FFold.append(FNames[round((N-1)*p):])

	# Check Algorithm Correctness (disjoint folds)
	for i in range(args.n_folds):
		for j in range(args.n_folds):
			if i==j: continue
			isdisjoint = set(FFold[i]).isdisjoint(set(FFold[j]))
			if not isdisjoint:
				print('PROBLEM: There is at least one couple of folds which are non-disjoint!!')
				exit(1)

	return XFold, YFold, FFold


def fold_generator(args, XFold, YFold, FFold):

	N=args.n_folds
	for fold_id in range(args.n_folds):

		X_test = XFold[fold_id] 	# e.g. fold_id = 0
		y_test = YFold[fold_id]
		F_test = FFold[fold_id]

		print('test_id:', fold_id, end = ' ')
		X_valid = XFold[(fold_id+1)%N]	# e.g. fold_id%N+1 = 1
		y_valid = YFold[(fold_id+1)%N]
		F_valid = FFold[(fold_id+1)%N]
		print('valid_id:', (fold_id+1)%N, end = ' ')

		X_train=[]
		y_train=[]
		F_train=[]
		print('train_ids:', end = ' ')
		for i in range(N-2):
			X_train+=XFold[(fold_id+i+2)%N]
			y_train+=YFold[(fold_id+i+2)%N]
			F_train+=FFold[(fold_id+i+2)%N]
			print((fold_id+i+2)%N, end = ' ')
		print()

		################### mingle train with valid sets ###########################

		def mingle_train_valid_set(args, Valid, Train):
			M = Train + Valid
			random.seed(args.seed)
			M = random.sample(M, len(M))

			p=args.n_folds*10 # NOTE: 10
			Valid_mingled = M[:p]
			Train_mingled = M[p:]
			return Valid_mingled, Train_mingled

		# merge train with valid
		X_valid, X_train = mingle_train_valid_set(args, X_valid, X_train)
		y_valid, y_train = mingle_train_valid_set(args, y_valid, y_train)
		F_valid, F_train = mingle_train_valid_set(args, F_valid, F_train)

		###################################################################


		# ################ Equal Class mingling ###################
		if args.particip_cross:

			X = X_train + X_valid
			y = y_train + y_valid
			F = F_train + F_valid
			idx_list = np.arange(len(y), dtype="int32")
			np.random.seed(args.seed)
			np.random.shuffle(idx_list)

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

		# ##########################################################################


		# ################ Inject new samples from test to train set ###################

		if args.n_injections > 0:
			Occ=[0]*10
			InjIdx=[]
			for i, t in enumerate(y):
				t = int(t)
				Occ[t] += 1
				if Occ[t] > args.n_injections: continue
				InjIdx.append(i)

			extra_X_train = [X_test[i] for i in range(len(X_test)) if i in InjIdx]
			extra_y_train = [y_test[i] for i in range(len(X_test)) if i in InjIdx]
			tmp_X_test  = [X_test[i] for i in range(len(X_test)) if i not in InjIdx]
			tmp_y_test  = [y_test[i] for i in range(len(X_test)) if i not in InjIdx]
			
			X_train += extra_X_train
			y_train += extra_y_train
			X_test = tmp_X_test
			y_test = tmp_y_test

		# ##########################################################################

		# print('train len:', len(X_train))
		# print('test len:', len(X_test))


		# Follow the torch.utils.data.Dataset prototype
		train_set = AirDataset(X_train, y_train, args.featrep)
		val_set = AirDataset(X_valid, y_valid, args.featrep)
		test_set = AirDataset(X_test, y_test, args.featrep)

		# Use torch.utils.data.Dataloader
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)

		# yield X_train, X_valid, X_test, y_train, y_valid, y_test, F_test
		yield train_loader, val_loader, test_loader, F_test



	########################################################################################

	# # Count Valid Recordings per Participant
	# Occ={'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0, 'G':0, 'H':0, 'I':0, 'J':0}
	# for filename in os.listdir(root):
	# 	Occ[filename.split('_')[0]] += 1

	# # Find the Minimum Number of Valid Recordings per Participant
	# k_min = min(Occ.keys(), key=(lambda k: Occ[k]))
	# min_participant = Occ[k_min]

	# # Gather All Recordings' Filenames per Participant
	# FDict = {'A':[], 'B':[], 'C':[], 'D':[], 'E':[], 'F':[], 'G':[], 'H':[], 'I':[], 'J':[]}
	# listdir = os.listdir(root)
	# shuffled_listdir = random.sample(listdir, len(listdir))
	# for filename in shuffled_listdir:
	# 	FDict[filename.split('_')[0]].append(filename)

	# # Count Valid Classes per Participant #TODO: and Create Equal Class Dataset
	# strToInt = {'zero':0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}
	# for k in FDict:
	# 	Occ=[0]*10
	# 	for filename in FDict[k]:
	# 		i = strToInt[filename.split('_')[1]]
	# 		Occ[i] += 1
		# print(Occ)


	# # Keep only the first #minim recordings per participant
	# MinimalFDict = {}
	# MinimalFList = []
	# for key in FDict:
	# 	MinimalFDict[key] = FDict[key][:minim]
	# 	MinimalFList += FDict[key][:minim]

	# print([ len(MinimalFDict[key]) for key in MinimalFDict ])

	########################################################################################


