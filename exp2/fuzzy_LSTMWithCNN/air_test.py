import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
# sys.path.append("../../")
from model import FUZZY
# sys.path.append("../")
from myutils import data_generator, fold_generator
from tools import plot_learning_curve, plot_accs, plot_confusion_matrix, str2bool
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
parser.add_argument('--layers', type=int, default=1,
					help='# of lstm layers')						
parser.add_argument('--bidirectional', type=str2bool, default=False)
parser.add_argument('--use_cnn', type=str2bool, default=False)
parser.add_argument('--train', type=str2bool, default=True)
# parser.add_argument('--model_name', type=str, default='fuzzy.pt')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')

parser.add_argument('--n_folds', default=10, type=int)
parser.add_argument('--cross_val', type=str2bool, default=True)
parser.add_argument('--model_dir', type=str, default='../../models/fuzzy/', help='../../models/sth')
parser.add_argument('--run_all_folds', type=str2bool, default=True)
parser.add_argument('--data_dir', type=str, default= '../../data/air_bulk_trails_all/')
parser.add_argument('--particip_cross', type=str2bool, default=True, help='participant cross validation') # NOTE

args = parser.parse_args()


# args.model_dir = args.model_dir+args.model+'/'


if args.run_all_folds:
	Nf=args.n_folds
else:
	Nf=0

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
		for static_data, trail_data, target, lengths in eval_loader:
			if args.cuda:
				static_data, trail_data, target = static_data.cuda(), trail_data.cuda(), target.cuda()               

			output = model(static_data.float(), trail_data.float(), lengths)
			eval_loss += F.nll_loss(output, target.long(), size_average=False).item()
			maxim, pred = output.data.max(1, keepdim=True)
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

			# Create Confusion Metrix
			rows = target.cpu().numpy()
			cols = pred.data.view_as(target).cpu().numpy()
			for c, (i,j) in enumerate(zip(rows, cols)):
				i, j = int(i), int(j)  
				cm[i,j] += 1
				if i!=j:
					# fimg = coorToImg(data[c].cpu().numpy())
					fimg = static_data[i].cpu().numpy()
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
	# for batch_idx, (data, target, lengths) in enumerate(train_loader):
	for batch_idx, (static_data, trail_data, target, lengths) in enumerate(train_loader):
		if args.cuda:
			static_data, trail_data, target = static_data.cuda(), trail_data.cuda(), target.cuda()

		# NOTE: data, target, lengths = Variable(data), Variable(target), Variable(lengths)

		optimizer.zero_grad()

		output = model(static_data.float(), trail_data.float(), lengths)
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


def run_epoch(epoch, XFold_s, YFold_s, FFold_s, XFold_t, YFold_t, FFold_t):
	global best_acc
	global mparams
	global min_loss
	print("************ EPOCH " + str(epoch) + " ************")

	for fold_id, (train_loader, val_loader, test_loader, F_test) in enumerate( fold_generator(args, XFold_s, YFold_s, FFold_s, XFold_t, YFold_t, FFold_t) ):
		
		if fold_id>Nf: continue 


		# if args.use_cnn:
		model_name = "fuzzy_"+str(fold_id)+".pt"
		
		print("************ Train_epoch_"+str(epoch)+"_fold_"+str(fold_id)+ " ************")
		train_loss = train(epoch, fold_id, train_loader)

		val_loss, val_acc, cm, FImgs, Pairs, Confidence = eval(val_loader, fold_id, name='Valid')

		print("************ Test_epoch_"+str(epoch)+"_fold_"+str(fold_id)+" ************")
		test_loss, test_acc, _, _, _, _ = eval(test_loader, fold_id, name='Test')

		# Save best model so far and Get Best Confusion Matrix
		# if val_acc>best_acc[fold_id]:
		if (val_loss < min_loss[fold_id] and val_acc == best_acc[fold_id]) or (val_acc > best_acc[fold_id]):
			model = mparams[fold_id]['model']

			best_acc[fold_id] = val_acc
			min_loss[fold_id] = val_loss	
			best_epoch[fold_id] = epoch

			with open(args.model_dir+model_name, "wb") as f:
				torch.save(model, f)


if __name__ == "__main__":

	# Dirs to store results
	try:
		os.mkdir(args.model_dir)
	except:
		print('No new model_dir was created. '+args.model_dir+'probably already exists.')

	imgs = './img/'
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

	# # NOTE:
	# today = date.today()	
	# model_name = "cnn.pt"

	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		if not args.cuda:
			print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# root = '../../data/air_bulk_trails/'
	root = args.data_dir

	if args.featrep=='identical':
		batch_size=1
	else:
		batch_size = args.batch_size

	n_classes = 10
	input_channels = 2
	hidden_dim = 300

	# Initialize n_fold Model Instances for Each Category: Visual, Audio, Fusion
	# use_cnn = (args.model=='cnn-lstm')
	mparams=[]
	lr = args.lr
	for i in range(args.n_folds):
		model = FUZZY(input_dim=input_channels, hidden_dim=hidden_dim, output_dim=n_classes, num_layers=args.layers, dropout=args.dropout, bidirectional=args.bidirectional, use_cnn=args.use_cnn, featrep=args.featrep)


		if args.cuda: 
			model.cuda()

		optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
		mparams += [{'model':model, 'lr':lr, 'optimizer':optimizer}]

	best_acc = [-0.1] * args.n_folds
	best_epoch = [-1] * args.n_folds
	min_loss = [1000000] * args.n_folds


	# (Pre-)Load the whole Dataset, equally segmented into non-overlapping sub-sets (i.e. folds).
	if args.particip_cross: # NOTE
		XFold_s, YFold_s, FFold_s = data_generator(args, root, featrep='image')
		XFold_t, YFold_t, FFold_t = data_generator(args, root, featrep='zeropad')	# else: #TODO
	# 	XFold, YFold, FFold = data_generator_bulk(args, root)

	# Train
	if args.train:
		for epoch in range(args.epochs):
			run_epoch(epoch, XFold_s, YFold_s, FFold_s, XFold_t, YFold_t, FFold_t)


	print('**************** Run Final Test *****************')

	sum_acc=0
	for fold_id, (train_loader, val_loader, test_loader, F_test) in enumerate( fold_generator(args, XFold_s, YFold_s, FFold_s, XFold_t, YFold_t, FFold_t) ):
		if not args.run_all_folds:
			if fold_id!=args.Nf: 
				continue 

		# model_name = args.model_dir+args.model+"_"+str(fold_id)+".pt"		

		model_name = args.model_dir+"fuzzy_"+str(fold_id)+".pt"

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


	# # Train
	# if args.train:
	# 	# TrainLoss, ValLoss, TrainAcc, ValAcc, best_cm, best_FImgs, best_Pairs, best_acc = train_model(train_loader, val_loader, max_epochs=args.epochs, input_dim=28, output_dim=10, dropout=0.7)
	# 	TrainLoss, ValLoss, TrainAcc, ValAcc = train_model(train_loader, val_loader, max_epochs=args.epochs, input_dim=2, hidden_dim=300, output_dim=10, num_layers=1, dropout=0.7, featrep=args.featrep)
	# 	# Learning Curve
	# 	# plt = plot_learning_curve(TrainLoss, ValLoss, range(1, len(ValLoss)+1))
	# 	plt = plot_learning_curve(TrainAcc, ValAcc, range(1, len(ValLoss)+1))
	# 	plt.savefig(imgs+'learning_curve.png')
	# 	plt.clf()
	# 	# plt.show()

	# # Load & Evaluate best model
	# start_time = time.time()
	# model = torch.load(open(model_name, "rb"))
	# print("--- %s seconds ---" % round((time.time() - start_time),2))	
	# test_loss, best_acc, best_cm, best_FImgs, best_Pairs, Confidence = eval_model(model, test_loader, name='TEST')

	# # Plot Confusion Matrix
	# pltcm = plot_confusion_matrix(best_cm, range(10),
	# 						normalize=False,
	# 						title='CNN Confusion matrix. Accuracy = ' + str(round(best_acc, 3)),
	# 						cmap=plt.cm.Blues)
	# pltcm.savefig(imgs+'best_cm.png')

	# # Plot mismatched samples
	# plt.clf()
	# for pair, img, conf in zip(best_Pairs, best_FImgs, Confidence):
	# 	plt.title("True Label: " + str(pair[0]) + "     Predicted:" + str(pair[1]) + "     Confidence:" + str(round(conf,3)))
	# 	plt.axis('off')
	# 	plt.imshow(img[:,::-1].T, cmap="gray")
	# 	plt.savefig(imgs+'mismatched_examples/'+'t'+ str(pair[0]) +'p'+ str(pair[1])+'.png')
	# 	# plt.show()
