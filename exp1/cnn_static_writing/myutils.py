import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import datasets, transforms
import csv
import random 

class FrameLevelDataset(Dataset):
	def __init__(self, feats, labels):
		self.lengths = [sample.shape[1] for sample in feats] # Find the lengths 

		self.feats = feats

		self.labels = labels

	def __getitem__(self, item):
		return self.feats[item], self.labels[item], self.lengths[item]

	def __len__(self):
		return len(self.feats)


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

def data_generator(root, batch_size):

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


	train_set = FrameLevelDataset(X_train, y_train)
	val_set = FrameLevelDataset(X_val, y_val)

	# Load Test
	X_test = []
	y_test = []
	for filename in os.listdir(root+'Test/'):
		npzfile = np.load(root+'Test/'+filename)
		feats = npzfile['input']
		img = coorToImg(feats)		
		X_test.append(img)
		y_test.append(npzfile['target'])

	test_set = FrameLevelDataset(X_test, y_test)

	# Use Dataloader
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

	return train_loader, val_loader, test_loader