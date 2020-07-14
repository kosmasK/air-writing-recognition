import matplotlib.pyplot as plt 
import numpy as np
import itertools



def checkforDuplicates(listOfElems):
	for idx, elem in enumerate(listOfElems):
		if listOfElems.count(elem) > 1:
			print(idx, elem)
			return True
	return False

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
	
def str2bool(v):
	if isinstance(v, bool):
	   return v
	if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	

def plot_learning_curve(train_scores, test_scores, train_sizes, ylim=(0.1, 1)):
	plt.figure()
	plt.title("Learning Curve")
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	plt.plot(train_sizes, train_scores, '-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores, '-', color="g",
			 label="Validation score")

	plt.legend(loc="best")
	return plt

def plot_accs(test_scores, train_sizes, ylim=(0, 1)):
	plt.figure()
	plt.title("Learning Curve")
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")

	plt.plot(train_sizes, train_scores, 'o-', color="r",
			 label="Training score")

	plt.legend(loc="best")
	return plt


def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	# plt.show()
	return plt