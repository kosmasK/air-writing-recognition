import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.fuzzy_LSTMWithCNN.model import FUZZY
from TCN.fuzzy_LSTMWithCNN.myutils import FrameLevelDataset, data_generator
from TCN.tools import plot_learning_curve, plot_accs, plot_confusion_matrix, str2bool
import matplotlib.pyplot as plt 
import numpy as np
import argparse
import itertools
import time
from torch import nn
import shutil
import torch
import os
from math import exp

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
					help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
					help='dropout applied to layers (default: 0.05)')					
parser.add_argument('--epochs', type=int, default=1000,
					help='upper epoch limit (default: 20)')
parser.add_argument('--lr', type=float, default=2e-3,
					help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
					help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed (default: 1111)')
parser.add_argument('--featrep', type=str, default='rescale',
					help='')
parser.add_argument('--bidirectional', type=str2bool, default=False)
parser.add_argument('--use_cnn', type=str2bool, default=True)
parser.add_argument('--train', type=str2bool, default=True)

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = '../../data/air_trails/'

batch_size = args.batch_size
n_classes = 10
input_channels = 2
epochs = args.epochs
steps = 0

lr = args.lr

def eval_model(model, eval_loader, epoch=None, name='VALIDATION'):
		FImgs=[]
		Pairs=[]
		Confidence=[]
		sum_loss = 0
		cm = np.zeros((10,10), dtype=int)
		eval_loss, count, correct = 0, 0, 0
		print
		with torch.set_grad_enabled(False):
			for static_data, trail_data, target, lengths in eval_loader:
				if args.cuda:
					static_data, trail_data, target = static_data.cuda(), trail_data.cuda(), target.cuda()               

				eval_predictions = model(static_data.float(), trail_data.float(), lengths) # the output for every batch's input containing the softmax probabilities
					
				sum_loss += loss_function(eval_predictions, target.long()) 
				maxim, index = eval_predictions.max(1) # index is nothing else thbut a list of the predicted_labels per batch. We collect the index with max softmax probability for every batch's input. This is our meaningful prediction.
				correct += torch.sum(index == target) # add the '1's to find the num of correct predictions
				count+=1

				# Confusion Matrix
				for i in range(len(static_data)):
					cm[target[i], index[i]] += 1 
					if target[i] != index[i]:
						# fimg = coorToImg(eval_feats[i].cpu().numpy())
						fimg = static_data[i].cpu().numpy()
						# FImgs += [eval_feats[i].cpu().numpy()]
						FImgs += [fimg]
						Pairs += [(target[i].item(), index[i].item())]
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


def train_model(train_loader, val_loader, max_epochs, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, bidirectional=False, use_cnn=True, featrep='rescale'):
	# Load Model
	global best_epoch, max_acc, best_cm
	global TrainLoss, ValLoss
	global model_name, loss_function
	
	# Load Model
	model = FUZZY(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, use_cnn=use_cnn, featrep=featrep) #TODO
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
	max_acc = 0
	min_loss = 100000

	epoch = 0
	# patience = 20
	# n_badMoves = 0
	# prev_loss = 100 # sth big
	TrainLoss, ValLoss, TrainAcc, ValAcc = [], [], [], []
	best_FImgs = []
	best_Pairs = []
	
	for epoch in range(max_epochs):

		# Training    
		sum_loss, count, correct = 0, 0, 0
		# for batch_feats, batch_labels, batch_lengths in train_loader:
		for static_data, trail_data, target, lengths in train_loader:
			if args.cuda:
				static_data, trail_data, target = static_data.cuda(), trail_data.cuda(), target.cuda()
			
			if args.cuda:
				model.cuda()
			
			model.zero_grad() #allways needed
			
			# Model's output
			predictions = model(static_data.float(), trail_data.float(), lengths)

			# Compute loss and back-propagate
			loss = loss_function(predictions, target.long()) 
			loss.backward()
			optimizer.step() 
			sum_loss += loss
			_, predicted_labels = predictions.max(1)
			correct += torch.sum(predicted_labels == target) # add the '1's to find the num of correct predictions
			count+=1			

		sum_loss = sum_loss.cpu()
		train_loss = sum_loss.detach().numpy()/count
		TrainLoss.append(train_loss)
		acc = correct.item() / len(train_loader.dataset)
		TrainAcc.append(acc)

		# Val
		val_loss, acc, cm, FImgs, Pairs, Confidence = eval_model(model, val_loader, epoch=epoch)
		ValLoss.append(val_loss)
		ValAcc.append(acc)   

		# Save best model
		# if (max_acc < acc):
		if (val_loss < min_loss):
			min_loss = val_loss
			max_acc = acc
			best_epoch = epoch
			best_cm = cm
			best_FImgs = FImgs
			best_Pairs = Pairs
			with open(model_name, "wb") as f:
				torch.save(model, f)
			
	print('Best epoch:', best_epoch, ', Validation Accuracy:', max_acc)

	return TrainLoss, ValLoss, TrainAcc, ValAcc#, best_cm, best_FImgs, best_Pairs, max_acc


if __name__ == "__main__":
	model_name = "lstm-cnn.pt"
	loss_function = nn.NLLLoss()

	# Load data
	train_loader, val_loader, test_loader = data_generator(root, batch_size, args.featrep)

	# Dir to store results
	imgs = './img/'
	try:
		os.mkdir(imgs+'mismatched_examples/')
	except:
		print('Error while creating img directory')

	try:
		shutil.rmtree(imgs+'mismatched_examples/')
	except:
		print('Error while deleting directory')
	os.mkdir(imgs+'mismatched_examples/')

	# Train
	if args.train:
		# TrainLoss, ValLoss, TrainAcc, ValAcc, best_cm, best_FImgs, best_Pairs, best_acc = train_model(train_loader, val_loader, max_epochs=args.epochs, input_dim=28, output_dim=10, dropout=0.7)
		TrainLoss, ValLoss, TrainAcc, ValAcc = train_model(train_loader, val_loader, max_epochs=args.epochs, input_dim=2, hidden_dim=300, output_dim=10, num_layers=1, dropout=0.7, featrep=args.featrep)
		# Learning Curve
		plt = plot_learning_curve(TrainAcc, ValAcc, range(1, len(ValLoss)+1))
		plt.savefig(imgs+'learning_curve.png')
		plt.clf()
		# plt.show()

	# Load & Evaluate best model
	start_time = time.time()
	model = torch.load(open(model_name, "rb"))
	print("--- %s seconds ---" % round((time.time() - start_time),2))	
	test_loss, best_acc, best_cm, best_FImgs, best_Pairs, Confidence = eval_model(model, test_loader, name='TEST')

	# Plot Confusion Matrix
	pltcm = plot_confusion_matrix(best_cm, range(10),
							normalize=False,
							title='CNN Confusion matrix. Accuracy = ' + str(round(best_acc, 3)),
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
