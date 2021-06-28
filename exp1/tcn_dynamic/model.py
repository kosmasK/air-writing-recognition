import torch.nn.functional as F
from torch import nn
import sys
# sys.path.append("../../")
from exp1.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, featrep):
        super(TCN, self).__init__()
        self.featrep = featrep
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs, lengths):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        if self.featrep=='zeropad':
            y1 = self.last_timestep(y1.transpose(1,2), lengths)
            o = self.linear(y1)
        elif self.featrep=='rescale':
            o = self.linear(y1[:, :, -1])
        
        return F.log_softmax(o, dim=1)

    def last_timestep(self, outputs, lengths):
        return self.last_by_index(outputs, lengths)

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                                outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()  