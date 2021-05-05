import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import sys
from model import LSTM, TCN, ConvNet
# sys.path.append("../")
from myutils import data_generator, fold_generator, data_generator_bulk # :
from tools import plot_learning_curve, plot_accs, plot_confusion_matrix, str2bool
import numpy as np
import argparse
import torch.nn as nn
import torch
import os
import shutil
from math import exp
import time 
from datetime import date

parser = argparse.ArgumentParser(description='python air_test.py --model tcn --levels 6') # TODO levels vs. layers
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
					help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05, #NOTE: maybe better of with no dropout
					help='dropout applied to layers (default: 0.05)')
parser.add_argument('--epochs', type=int, default=1000,
					help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
					help='kernel size (default: 7)')					
parser.add_argument('--lr', type=float, default=2e-3,
					help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
					help='optimizer to use (default: Adam)')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed (default: 1111)')
parser.add_argument('--featrep', type=str, default='zeropad',
					help='rescale, zeropad, identical')
parser.add_argument('--levels', type=int, default=6,
					help='# of tcn levels')			
parser.add_argument('--layers', type=int, default=1,
					help='# of lstm layers')	
parser.add_argument('--nhid', type=int, default=25,
					help='number of hidden units per tcn layer (default: 25)')									
parser.add_argument('--bidirectional', type=str2bool, default=False)
parser.add_argument('--use_cnn', type=str2bool, default=True)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--model_name', type=str, default='lstm.pt')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
					help='report interval')

parser.add_argument('--n_folds', default=10, type=int)
parser.add_argument('--Nf', type=int, default=0, help='folder id to run if particip_cross is False')
parser.add_argument('--model_dir', type=str, default='../models/', help='../../models/sth')
parser.add_argument('--run_all_folds', type=str2bool, default=True)
parser.add_argument('--data_dir', type=str, default= '../data/air_bulk_trails_all/')
parser.add_argument('--particip_cross', type=str2bool, default=True, help='participant cross validation')
parser.add_argument('--n_injections', type=int, default=0, help='number of injected samples per class from test to train') 

parser.add_argument('--model', type=str, default='', help='cnn, cnn-lstm, lstm, tcn_dynamic, tcn_static')

args = parser.parse_args()

if args.model not in ["cnn", "cnn-lstm", "tcn_dynamic", "lstm", "tcn_static"]:
	print('Please choose a correct model (cnn", "cnn-lstm", "tcn_dynamic", "tcn_static", "lstm") to be employed using the argument --model.')
	exit(1)

if args.model == 'cnn':
	args.featrep = 'image'


# if args.model == 'cnn-lstm':
# 	args.featrep = 'rescale'

if args.model == 'tcn_static':
	args.featrep = 'static'


args.model_dir = args.model_dir+args.model+'/'

if args.run_all_folds:
	args.Nf=args.n_folds
# else:
# 	args.Nf=0

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


def eval(eval_loader, fold_id, name='Validation'):
	global mparams

	model = mparams[fold_id]['model']

	FImgs=[]
	Pairs=[]
	Confidence=[]	
	model.eval()
	eval_loss = 0
	correct = 0
	cm = np.zeros((10,10), dtype=int)
	with torch.no_grad():
		for data, target, lengths in eval_loader:
			if args.cuda:
				data, target, lengths = data.cuda(), target.cuda(), lengths.cuda()

			output = model(data.float(), lengths)
			eval_loss += F.nll_loss(output, target.long(), size_average=False).item()
			maxim, pred = output.data.max(1, keepdim=True)
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

			# Create Confusion Metrix
			rows = target.cpu().numpy()
			cols = pred.data.view_as(target).cpu().numpy()
			for c, (i,j) in enumerate(zip(rows, cols)):
				i, j = int(i), int(j) # NOTE: maybe need to 
				cm[i,j] += 1
				if i!=j:
					# fimg = coorToImg(data[c].cpu().numpy())	
					if args.model=="cnn" or args.model=="tcn_static":
						fimg = data[i].cpu().numpy()
					else:
						fimg = coorToImg(data[c].cpu().numpy())
					FImgs += [fimg]
					Pairs += [(i, j)]
					Confidence += [exp(maxim[c].item())]

		eval_loss /= len(eval_loader.dataset)
		print('Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(			
			eval_loss, correct, len(eval_loader.dataset),
			100. * correct / len(eval_loader.dataset)))

		eval_acc = 100. * correct / len(eval_loader.dataset)

		return eval_loss, eval_acc, cm, FImgs, Pairs, Confidence


def train(ep, fold_id, train_loader):
	global mparams

	train_loss, count, correct = 0, 0, 0

	optimizer = mparams[fold_id]['optimizer']
	model = mparams[fold_id]['model']
	
	model.train()
	for batch_idx, (data, target, lengths) in enumerate(train_loader):
		if args.cuda: data, target, lengths = data.cuda(), target.cuda(), lengths.cuda()

		data, target, lengths = Variable(data), Variable(target), Variable(lengths)

		optimizer.zero_grad()

		output = model(data.float(), lengths)
		loss = F.nll_loss(output, target.long())
		loss.backward()

		optimizer.step()
		train_loss += loss

		if batch_idx > 0 and batch_idx % args.log_interval == 0:
			# print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				ep, batch_idx * batch_size, len(train_loader.dataset),
				100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval))
			train_loss = 0

	mparams[fold_id]['optimizer'] = optimizer # 


	return train_loss.item()/len(train_loader.dataset)


def run_epoch(epoch, XFold, YFold, FFold):
	global best_acc
	global mparams
	global min_loss
	print("************ EPOCH " + str(epoch) + " ************")

	for fold_id, (train_loader, val_loader, test_loader, F_test) in enumerate( fold_generator(args, XFold, YFold, FFold) ):
		
		if not args.run_all_folds:
			if fold_id!=args.Nf: 
				continue 

		if args.n_injections>0:
			model_name = args.model+"_"+str(fold_id)+'_'+str(args.n_injections)+".pt"
		else:
			model_name = args.model+"_"+str(fold_id)+".pt"
		
		print("************ Train_epoch_"+str(epoch)+"_fold_"+str(fold_id)+ " ************")
		train_loss = train(epoch, fold_id, train_loader)

		val_loss, val_acc, cm, FImgs, Pairs, Confidence = eval(val_loader, fold_id, name='Valid')

		print("************ Test_epoch_"+str(epoch)+"_fold_"+str(fold_id)+" ************")
		test_loss, test_acc, _, _, _, _ = eval(test_loader, fold_id, name='Test')

		# Save best model so far and Get Best Confusion Matrix
		
		# if val_acc>best_acc[fold_id]:
		if (val_loss < min_loss[fold_id] and val_acc == best_acc[fold_id]) or (val_acc > best_acc[fold_id]):
		# if val_loss < min_loss[fold_id] or val_acc > best_acc[fold_id]:
			model = mparams[fold_id]['model']

			best_acc[fold_id] = val_acc
			min_loss[fold_id] = val_loss	
			best_epoch[fold_id] = epoch

			with open(args.model_dir+model_name, "wb") as f:
				torch.save(model, f)


if __name__ == "__main__":

	# Dirs to store results
	try:
		os.mkdir('../models/')
		os.mkdir(args.model_dir)
	except:
		print('No new model_dir was created. '+args.model_dir+' probably already exists.')


	try:
		os.mkdir(args.model)
	except:
		print('Failed to create model dir. Probaby already exists.')

	imgs = './'+args.model+'/img/'
	try:
		os.mkdir(imgs)
	except:
		print('Failed to create ./img/ dir. Probaby already exists.')

	try:
		shutil.rmtree(imgs+'mismatched_examples/')
	except:
		print("Failed to delete mismatched_examples directory. Probably it didn't even exist.")
	os.mkdir(imgs+'mismatched_examples/')

	for fold_id in range(args.n_folds):
		try:
			os.mkdir(imgs+'mismatched_examples/'+str(fold_id)+'/')
		except:
			print('Failed to create mismatched_examples/ subdirectory '+imgs+'mismatched_examples/'+str(fold_id)+'/')
			continue

	model_name = args.model+"_"+str(fold_id)+".pt"

	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		if not args.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	root = args.data_dir

	if args.featrep=='identical':
		batch_size=1
	else:
		batch_size = args.batch_size

	n_classes = 10
	input_channels = 2
	if args.model == 'tcn_static': 
		input_channels = 1
	hidden_dim = 300
	channel_sizes = [args.nhid] * args.levels
	kernel_size = args.ksize

	# Initialize n_fold Model Instances for Each Category: Visual, Audio, Fusion
	use_cnn = (args.model=='cnn-lstm')
	mparams=[]
	lr = args.lr
	for i in range(args.n_folds):
		# model = TCN(input_channels, n_classes, channel_sizes, kernel_size, dropout=args.dropout, featrep=args.featrep)
		if args.model in ['tcn_dynamic', 'tcn_static']:
			model = TCN(input_channels, n_classes, channel_sizes, kernel_size, dropout=args.dropout, featrep=args.featrep)
		elif args.model in ['lstm', 'cnn-lstm']:
			model = LSTM(input_dim=input_channels, hidden_dim=hidden_dim, output_dim=n_classes, num_layers=args.layers, dropout=args.dropout, bidirectional=args.bidirectional, use_cnn=use_cnn, featrep=args.featrep)
		elif args.model == 'cnn':
			model = ConvNet(input_dim=input_channels, output_dim=n_classes, dropout=0.0, device=args.cuda)

		if args.cuda: 
			model.cuda()

		optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
		mparams += [{'model':model, 'lr':lr, 'optimizer':optimizer}]

	best_acc = [-0.1] * args.n_folds
	best_epoch = [-1] * args.n_folds
	min_loss = [1000000] * args.n_folds

	if args.n_injections>0:
		model_name = args.model_dir+args.model+"_"+str(args.Nf)+".pt"		
		model = torch.load(open(model_name, "rb"))
		mparams[fold_id]['model'] = model


	# (Pre-)Load the whole Dataset, equally segmented into non-overlapping sub-sets (i.e. folds).

	if args.particip_cross: # NOTE
		XFold, YFold, FFold = data_generator(args, root)
	else:
		XFold, YFold, FFold = data_generator_bulk(args, root)

	# Train
	if args.train:
		for epoch in range(args.epochs):
			run_epoch(epoch, XFold, YFold, FFold)

	print('**************** Run Final Test *****************')

	sum_acc=0
	for fold_id, (train_loader, val_loader, test_loader, F_test) in enumerate( fold_generator(args, XFold, YFold, FFold) ):
		if not args.run_all_folds:
			if fold_id!=args.Nf: 
				continue 

		model_name = args.model_dir+args.model+"_"+str(fold_id)+".pt"		
		if args.n_injections>0:
			model_name = args.model_dir+args.model+"_"+str(fold_id)+'_'+str(args.n_injections)+".pt"	
			print(model_name)	

		print('-' * 89)
		print("Fold_"+str(fold_id))
		model = torch.load(open(model_name, "rb"))
		mparams[fold_id]['model'] = model
		test_loss, test_acc, cm, FImgs, Pairs, Confidence = eval(test_loader, fold_id, name='Test')

		if args.train: 
			# print("Min Loss:", min_loss)
			print("Tets acc:", test_acc)
			print('Best epoch:', best_epoch[fold_id])

		# Plot Confusion Matrix
		pltcm = plot_confusion_matrix(cm, range(10),
								normalize=False,
								title='Validation Confusion matrix',
								cmap=plt.cm.Blues)

		pltcm.savefig(imgs+'best_cm_fold_'+str(fold_id)+'.png')

		plt.clf()
		for pair, img, conf in zip(Pairs, FImgs, Confidence):
			# plt.title("True Label: " + str(pair[0]) + "     Predicted:" + str(pair[1]) + "     Confidence:" + str(round(conf,3)))
			plt.axis('off')
			plt.imshow(img[:,::-1].T, cmap="gray")
			plt.savefig(imgs+'mismatched_examples/'+str(fold_id)+'/'+'t'+ str(pair[0]) +'p'+ str(pair[1])+'.png')

		sum_acc += test_acc

	print('-' * 89)
	print()
	avg_acc = sum_acc/args.n_folds

	print('Average accuracy: {:.1f}%\n'.format(sum_acc/args.n_folds))


