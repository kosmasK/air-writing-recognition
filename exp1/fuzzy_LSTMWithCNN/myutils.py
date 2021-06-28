import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import datasets, transforms
import csv
import random 
from exp1.tools import coorToImg, do_the_scaling
from skimage.transform import rescale


class FrameLevelDataset(Dataset):
	def __init__(self, static_feats, trail_feats, labels, featrep):
		self.labels = labels

		# Static
		self.static_feats = static_feats

		# Trail
		self.lengths = [sample.shape[1] for sample in trail_feats] # Find the lengths 
		if featrep=='rescale':
			self.trail_feats = self.rescale(trail_feats)
		elif featrep=='zeropad':
			self.trail_feats = self.zero_pad_and_stack(trail_feats)
		elif featrep=='identical':
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


def static_data_generator(root, batch_size):
	global shuffle
	# Load Train 
	X=[]
	y=[]
	for filename in os.listdir(root+'Train/'):
		npzfile = np.load(root+'Train/'+filename)
		feats = npzfile['input']
		img = coorToImg(feats)
		X.append(img)
		y.append(npzfile['target'])

	val_ids = [628, 693, 699, 657, 594, 252, 849, 787, 36, 632, 290, 744, 143, 197, 665, 106, 801, 210, 414, 380, 335, 922, 484, 145, 151, 400, 
			3, 916, 793, 394, 342, 166, 155, 824, 621, 941, 174, 645, 920, 715, 6, 561, 135, 11, 358, 777, 88, 0, 291, 80, 57, 24, 909, 54, 626, 
			866, 647, 757, 590, 107, 312, 224, 323, 303, 62, 845, 45, 897, 43, 864, 499, 137, 82, 144, 842, 734, 496, 359, 662, 435, 675, 470, 788, 
			325, 743, 636, 473, 331, 40, 94, 328, 798, 614, 794, 272, 430, 653, 100, 534, 930]

	train_ids = [i for i in range(len(X)) if not i in val_ids]

	X_train = [X[i] for i in train_ids]
	y_train = [y[i] for i in train_ids]
	X_val   = [X[i] for i in val_ids]
	y_val   = [y[i] for i in val_ids]

	# Load Test
	X_test = []
	y_test = []
	for filename in os.listdir(root+'Test/'):
		npzfile = np.load(root+'Test/'+filename)
		feats = npzfile['input']
		img = coorToImg(feats)		
		X_test.append(img)
		y_test.append(npzfile['target'])

	return X_train, y_train, X_val, y_val, X_test, y_test


def trail_data_generator(root, batch_size, featrep):
	global shuffle
	# Load Train 
	X=[]
	y=[]
	for filename in os.listdir(root+'Train/'):
		npzfile = np.load(root+'Train/'+filename)
		X.append(npzfile['input'])
		y.append(npzfile['target'])

	val_ids = [628, 693, 699, 657, 594, 252, 849, 787, 36, 632, 290, 744, 143, 197, 665, 106, 801, 210, 414, 380, 335, 922, 484, 145, 151, 400, 
			3, 916, 793, 394, 342, 166, 155, 824, 621, 941, 174, 645, 920, 715, 6, 561, 135, 11, 358, 777, 88, 0, 291, 80, 57, 24, 909, 54, 626, 
			866, 647, 757, 590, 107, 312, 224, 323, 303, 62, 845, 45, 897, 43, 864, 499, 137, 82, 144, 842, 734, 496, 359, 662, 435, 675, 470, 788, 
			325, 743, 636, 473, 331, 40, 94, 328, 798, 614, 794, 272, 430, 653, 100, 534, 930]

	train_ids = [i for i in range(len(X)) if not i in val_ids]

	X_train = [X[i] for i in train_ids]
	y_train = [y[i] for i in train_ids]
	X_val   = [X[i] for i in val_ids]
	y_val   = [y[i] for i in val_ids]


	# Load Test
	X_test=[]
	y_test=[]
	for filename in os.listdir(root+'Test/'):
		npzfile = np.load(root+'Test/'+filename)
		X_test.append(npzfile['input'])
		y_test.append(npzfile['target'])
	return X_train, y_train, X_val, y_val, X_test, y_test



def data_generator(root, batch_size, featrep):
	X_static_train, y_static_train, X_static_val, y_val, X_static_test, y_test = static_data_generator(root, batch_size)
	X_trail_train, y_trail_train, X_trail_val, _, X_trail_test, _ = trail_data_generator(root, batch_size, featrep)

	# Just Checking 
	if not y_static_train==y_trail_train:
		print('Error when shuffling!')
		exit

	y_train = y_static_train

	train_set = FrameLevelDataset(X_static_train, X_trail_train, y_train, featrep)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

	val_set = FrameLevelDataset(X_static_val, X_trail_val, y_val, featrep)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

	test_set = FrameLevelDataset(X_static_test, X_trail_test, y_test, featrep)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

	return train_loader, val_loader, test_loader