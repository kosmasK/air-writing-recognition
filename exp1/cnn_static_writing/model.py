import torch.nn.functional as F
from torch import nn
import sys
from exp1.tcn import TemporalConvNet
import torch.nn.functional as F

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
	def __init__(self, input_dim, output_dim, num_classes=10, dropout=0.0, device='cuda'):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.fc1 = nn.Linear( 32*5*5, 100, bias=False)
		self.fc2 = nn.Linear(100, num_classes, bias=False)
		
	def forward(self, x):
		x = x.transpose(1,2).unsqueeze(1) # Turn (batch_size x seq_len x input_size) into (batch_size x input_size x seq_len) for CNN 
		out = self.layer1(x)
		out = self.layer2(out)
		
		out = out.reshape(out.size(0), -1)

		out = F.relu(self.fc1(out))
		out = self.fc2(out)

		return F.log_softmax(out, dim=1)

