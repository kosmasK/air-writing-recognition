import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import datasets, transforms
from skimage.transform import rescale

class FrameLevelDataset(Dataset):
	def __init__(self, x, labels, featrep):
		self.lengths = [sample.shape[1] for sample in x] # Find the lengths 

		if featrep=='rescale':
			self.x = self.rescale(x)
		elif featrep=='zeropad':
			self.x = self.zero_pad_and_stack(x)
		elif featrep=='identical':
			self.x = x

		self.labels = labels

	def rescale(self, x):
		maxLen = max(self.lengths)
		print(type(x), len(x), x[0].shape)
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


def data_generator(root, batch_size, featrep):

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

	train_set = FrameLevelDataset(X_train, y_train, featrep)
	val_set = FrameLevelDataset(X_val, y_val, featrep)

	# Load Test
	X_test=[]
	y_test=[]
	for filename in os.listdir(root+'Test/'):
		npzfile = np.load(root+'Test/'+filename)
		X_test.append(npzfile['input'])
		y_test.append(npzfile['target'])
		
	test_set = FrameLevelDataset(X_test, y_test, featrep)

	# Use Dataloader
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

	return train_loader, val_loader, test_loader