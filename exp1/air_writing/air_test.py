import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import sys
sys.path.append("../../")
from TCN.air_writing.model import TCN
from TCN.air_writing.myutils import FrameLevelDataset, data_generator
from TCN.tools import plot_learning_curve, plot_accs, plot_confusion_matrix, str2bool
import numpy as np
import argparse
from math import exp
import os
import shutil
import time

parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
					help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
					help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
					help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
					help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=100,
					help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
					help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=6,
					help='# of levels ')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
					help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
					help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
					help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
					help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
					help='random seed (default: 1111)')
parser.add_argument('--featrep', type=str, default='rescale',
					help="'rescale', 'zeropad', 'identical'")
parser.add_argument('--train', type=str2bool, default=True)

# python air_test.py --featrep zeropad 
# Namespace(batch_size=64, clip=-1, cuda=True, dropout=0.05, epochs=100, featrep='zeropad', ksize=7, levels=6, log_interval=40, lr=0.002, nhid=150, optim='Adam', seed=1111, train=True)


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

print(args)
train_loader, val_loader, test_loader = data_generator(root, batch_size, args.featrep)

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout, featrep=args.featrep)

if args.cuda:
	model.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


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

def train(ep):
	global steps
	train_loss = 0
	model.train()
	for batch_idx, (data, target, lengths) in enumerate(train_loader):
		if args.cuda: data, target, lengths = data.cuda(), target.cuda(), lengths.cuda()

		data, target, lengths = Variable(data), Variable(target), Variable(lengths)
		optimizer.zero_grad()
		optimizer.zero_grad()
		output = model(data.float(), lengths)
		loss = F.nll_loss(output, target.long())
		loss.backward()

		if args.clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
		optimizer.step()
		train_loss += loss
		if batch_idx > 0 and batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				ep, batch_idx * batch_size, len(train_loader.dataset),
				100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
			train_loss = 0

	return train_loss.item()/len(train_loader.dataset)


def eval(eval_loader, name='Validation'):
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
			# data, target, lengths = Variable(data, volatile=True), Variable(target), Variable(lengths)
			output = model(data.float(), lengths)
			eval_loss += F.nll_loss(output, target.long(), size_average=False).item()
			# pred = output.data.max(1, keepdim=True)[1]
			maxim, pred = output.data.max(1, keepdim=True)
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

			# Create Confusion Metrix
			rows = target.cpu().numpy()
			cols = pred.data.view_as(target).cpu().numpy()
			for c, (i,j) in enumerate(zip(rows, cols)):
				cm[i,j] +=1
				if i!=j:
					fimg = coorToImg(data[c].cpu().numpy())	
					FImgs += [fimg]
					Pairs += [(i, j)]
					Confidence += [exp(maxim[c].item())]

		eval_loss /= len(eval_loader.dataset)
		# print(name, 'SET: Epoch:', epoch)
		print('Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(			
			eval_loss, correct, len(eval_loader.dataset),
			100. * correct / len(eval_loader.dataset)))

		eval_acc = 100. * correct / len(eval_loader.dataset)
		# print(len(eval_loader.dataset))

		return eval_loss, eval_acc, cm, FImgs, Pairs, Confidence



if __name__ == "__main__":
	model_name = "tcn_dynamic.pt"
	
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

	maximum = 0
	min_loss = 100000

	train_losses, test_losses, val_losses, test_scores, val_scores = [], [], [], [], []
	if args.train:
		for epoch in range(1, epochs+1):
			print("************ Train ************")
			train_loss = train(epoch)
			train_losses.append(train_loss)

			# val_loss, val_acc, cm, FImgs, Pairs, Confidence = eval(train_loader, name='Train')


			val_loss, val_acc, cm, FImgs, Pairs, Confidence = eval(val_loader, name='Valid')
			val_losses.append(val_loss)
			val_scores.append(val_acc)	
			# print(type(val_loss))

			print("************ Test ************")
			test_loss, test_acc, _, _, _, _ = eval(test_loader, name='Test')

			test_losses.append(test_loss)
			test_scores.append(test_acc)

			# Get Best Confusion Matrix
			if val_acc>maximum:
			# if (val_loss < min_loss):
				maximum=test_acc
				min_loss = val_loss	
				best_epoch = epoch
				# best_cm = cm
				# best_FImgs = FImgs
				# best_Pairs = Pairs
				with open(model_name, "wb") as f:
					torch.save(model, f)

	# Load & Test best model
	start_time = time.time()
	model = torch.load(open(model_name, "rb"))
	print('************ Final Test ****************')
	if args.train: 
		print("Min Loss:", min_loss)
		print('Best epoch:', best_epoch)
	print("--- %s seconds ---" % round((time.time() - start_time),2))	
	val_loss, val_acc, cm, FImgs, Pairs, Confidence = eval(test_loader, name='Test')

	# Plot Confusion Matrix
	pltcm = plot_confusion_matrix(cm, range(10),
							normalize=False,
							title='Validation Confusion matrix',
							cmap=plt.cm.Blues)
	pltcm.savefig(imgs+'best_cm.png')

	# Plot mismatched samples
	plt.clf()
	for pair, img, conf in zip(Pairs, FImgs, Confidence):
		# plt.title("True Label: " + str(pair[0]) + "     Predicted:" + str(pair[1]) + "     Confidence:" + str(round(conf,3)))
		plt.axis('off')
		plt.imshow(img[:,::-1].T, cmap="gray")
		plt.savefig(imgs+'mismatched_examples/'+'t'+ str(pair[0]) +'p'+ str(pair[1])+'.png')
		# plt.show()
