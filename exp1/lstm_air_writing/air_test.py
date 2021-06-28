import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import sys
sys.path.append("../../")
from model import LSTM
from myutils import FrameLevelDataset, data_generator
from exp1.tools import plot_learning_curve, plot_accs, plot_confusion_matrix, str2bool
import numpy as np
import argparse
import torch.nn as nn
import torch
import os
import shutil
from math import exp
import time 
from datetime import date

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
					help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05, #NOTE: maybe better of with no dropout
					help='dropout applied to layers (default: 0.05)')
parser.add_argument('--epochs', type=int, default=1000,
					help='upper epoch limit (default: 20)')
parser.add_argument('--lr', type=float, default=2e-3,
					help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
					help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed (default: 1111)')
parser.add_argument('--featrep', type=str, default='rescale')
					# help='rescale', 'zeropad', 'identical')
parser.add_argument('--levels', type=int, default=1,
					help='# of levels')					
parser.add_argument('--bidirectional', type=str2bool, default=False)
parser.add_argument('--use_cnn', type=str2bool, default=True)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--model_name', type=str, default='lstm.pt')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = '../../data/air_trails/'

if args.featrep=='identical':
	batch_size=1
else:
	batch_size = args.batch_size

n_classes = 10
input_channels = 2
steps = 0

print(args)


def coorToImg(feats):
	w, h = 28, 28
	# NORMALIZE (AGAIN)
	minXY = np.amin(feats, axis=1)
	maxXY = np.amax(feats, axis=1)
	normXY = np.array([do_the_scaling(xy, minXY, maxXY) for xy in feats.T])

	# FILL IMG ARRAY 
	img = np.ones((w, h))
	normXY *= w-1
	normXY = normXY.astype(int)
	for xy in normXY:
		# Img
		row, col = xy[0], xy[1]
		img[row, col] = 0

	return img

def do_the_scaling(vector, minim, maxim):
	numerator = vector - minim
	denominator = maxim - minim
	return numerator/denominator

def eval_model(model, eval_loader, epoch=None, name='VALIDATION'):
		FImgs=[]
		Pairs=[]
		Confidence=[]
		sum_loss = 0
		cm = np.zeros((10,10), dtype=int)
		eval_loss, count, correct = 0, 0, 0
		print
		with torch.set_grad_enabled(False):
			for eval_feats, eval_labels, eval_lengths in eval_loader:
				if args.cuda:
					eval_feats = eval_feats.cuda()
					eval_lengths = eval_lengths.cuda()
					eval_labels = eval_labels.cuda()                

				eval_predictions = model(eval_feats.float(), eval_lengths) # the output for every batch's input containing the softmax probabilities
					
				sum_loss += loss_function(eval_predictions, eval_labels.long()) 
				maxim, index = eval_predictions.max(1) # index is nothing else thbut a list of the predicted_labels per batch. We collect the index with max softmax probability for every batch's input. This is our meaningful prediction.
				correct += torch.sum(index == eval_labels) # add the '1's to find the num of correct predictions
				count+=1

				# Confusion Matrix
				for i in range(len(eval_feats)):
					cm[eval_labels[i], index[i]] += 1 
					if eval_labels[i] != index[i]:
						fimg = coorToImg(eval_feats[i].cpu().numpy())
						# FImgs += [eval_feats[i].cpu().numpy()]
						FImgs += [fimg]
						Pairs += [(eval_labels[i].item(), index[i].item())]
						Confidence += [exp(maxim[i].item())]

			eval_loss = sum_loss.cpu().detach().numpy() / count 
			acc = correct.item() / len(eval_loader.dataset)

			##################################
			if epoch: print(name, 'SET: Epoch:', epoch)
			print('Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
				eval_loss, correct, len(eval_loader.dataset),
				100. * correct / len(eval_loader.dataset)))				
			##################################
		
		return eval_loss, acc, cm, FImgs, Pairs, Confidence


def train_model(max_epochs, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, bidirectional=False, use_cnn=True, featrep='rescale'):

	# Load Model
	global best_epoch, max_acc, best_cm
	global TrainLoss, ValLoss
	global model_name, loss_function
	# Load Model
	model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, use_cnn=use_cnn, featrep=args.featrep)
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
	min_loss = 100000
	max_acc = 0
	epoch = 0

	TrainLoss, ValLoss, TrainAcc, ValAcc = [], [], [], []
	best_FImgs = []
	best_Pairs = []
	
	for epoch in range(max_epochs):

		# Training    
		sum_loss, count, correct = 0, 0, 0
		for batch_feats, batch_labels, batch_lengths in train_loader:
			
			if args.cuda:
				model.cuda()
			
			model.zero_grad() #allways needed
			
			if args.cuda:
				batch_feats = batch_feats.cuda() # not really needed
				batch_lengths = batch_lengths.cuda()
				batch_labels = batch_labels.cuda()

			# Model's output
			predictions = model(batch_feats.float(), batch_lengths) # float() is used to dtype bug

			# Compute loss and back-propagate
			loss = loss_function(predictions, batch_labels.long()) 
			loss.backward()
			optimizer.step() 
			sum_loss += loss
			_, predicted_labels = predictions.max(1)
			correct += torch.sum(predicted_labels == batch_labels) # add the '1's to find the num of correct predictions
			count+=1			

		sum_loss = sum_loss.cpu()
		train_loss = sum_loss.detach().numpy()/count
		TrainLoss.append(train_loss)
		acc = correct.item() / len(train_loader.dataset)
		TrainAcc.append(acc)

		# Validation
		val_loss, acc, cm, FImgs, Pairs, Confidence = eval_model(model, val_loader, epoch=epoch)
		ValLoss.append(val_loss)
		ValAcc.append(acc)

		# # Early stopping 
		# if (prev_loss < val_loss):
		# 	n_badMoves += 1
		# else:
		# 	n_badMoves = 0
			
		# if n_badMoves >= patience:
		# 	print('Stopped training at epoch:', epoch, ', loss:', loss.cpu().detach().numpy())
		# 	break
		# prev_loss = val_loss        

		# Save best model
		# if (max_acc < acc) or (max_acc == acc and val_loss < min_loss):
		if (val_loss < min_loss):
			min_loss = val_loss
			max_acc = acc
			best_epoch = epoch
			best_cm = cm
			best_FImgs = FImgs
			best_Pairs = Pairs
			# today.strftime("%b-%d-%Y")
			# time.time()
			with open(model_name, "wb") as f:
				torch.save(model, f)

	print('Best epoch:', best_epoch, ', Validation Accuracy:', max_acc)

	return TrainLoss, ValLoss, TrainAcc, ValAcc, best_cm, best_FImgs, best_Pairs, max_acc

if __name__ == "__main__":
	today = date.today()	
	if args.use_cnn:
		model_name = "cnn-"+args.model_name
	else:
		model_name = args.model_name

	loss_function = nn.NLLLoss()

	# Load data
	train_loader, val_loader, test_loader = data_generator(root, batch_size, args.featrep)

	# Dir to store results
	imgs = './img/'
	try:
		os.mkdir(imgs)
	except:
		print('Error while creating img directory')

	try:
		shutil.rmtree(imgs+'mismatched_examples/')
	except:
		print('Error while deleting directory')
	os.mkdir(imgs+'mismatched_examples/')


	# Train
	if args.train:
		TrainLoss, ValLoss, TrainAcc, ValAcc, best_cm, best_FImgs, best_Pairs, best_acc = train_model(max_epochs=args.epochs, input_dim=2, hidden_dim=300, output_dim=10, 
																										num_layers=args.levels, dropout=args.dropout, bidirectional=args.bidirectional, 
																										use_cnn=args.use_cnn, featrep=args.featrep)
		# Learning Curveb
		plt = plot_learning_curve(TrainAcc, ValAcc, range(1, len(ValLoss)+1))
		plt.savefig(imgs+'learning_curve.png')
		plt.clf()
		# plt.show()

	# Load & Test best model
	start_time = time.time()
	model = torch.load(open(model_name, "rb"))
	print("--- %s seconds ---" % round((time.time() - start_time),2))	
	test_loss, best_acc, best_cm, best_FImgs, best_Pairs, Confidence = eval_model(model, test_loader, epoch=None, name='Test')

	# Plot Confusion Matrix
	pltcm = plot_confusion_matrix(best_cm, range(10),
							normalize=False,
							title='LSTM Confusion matrix. Accuracy = ' + str(round(best_acc, 3)),
							cmap=plt.cm.Blues)
	pltcm.savefig(imgs+'best_cm.png')

	# Plot mismatched samples
	plt.clf()
	for pair, img, conf in zip(best_Pairs, best_FImgs, Confidence):
		plt.title("True Label: " + str(pair[0]) + "     Predicted:" + str(pair[1]) + "     Confidence:" + str(round(conf,3)))
		plt.axis('off')
		plt.imshow(img[:,::-1].T, cmap="gray")
		plt.savefig(imgs+'mismatched_examples/'+'t'+ str(pair[0]) +'p'+ str(pair[1])+'.png')
		# plt.show()

