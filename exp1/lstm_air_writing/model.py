import torch.nn.functional as F
from torch import nn
import sys
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.0, bidirectional=False, use_cnn=True, featrep='rescale'):
        super(LSTM, self).__init__()

        self.featrep = featrep
        self.bidirectional = bidirectional
        self.use_cnn = use_cnn
        self.feature_size = hidden_dim * 2 if self.bidirectional else hidden_dim

        
        if self.use_cnn:  
            self.c1 = nn.Conv1d(input_dim, hidden_dim, 1)
            self.p1 = nn.AvgPool1d(2)
        
        # Initialize the LSTM, Dropout, Output layers    
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers      
        if self.use_cnn:  #TODO: shorten expression
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.softmax = nn.LogSoftmax()
        
    def init_hidden(self, batch_size):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)        
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())

        
    def forward(self, x, lengths):
        
        batch_size = lengths.shape[0]  
        
        hidden = self.init_hidden(batch_size)
        # TODO: what if featrep == rescale ()
        if self.use_cnn: # In case we use cnn
            c = self.c1(x)
            lengths = lengths-1

            lstm_out, hidden = self.lstm(c.transpose(1,2), hidden) 
        else:
            lstm_out, hidden = self.lstm(x.transpose(1,2), hidden) # remember that: batch_first=True (check init_hidden). Also, lstm_out.size() = x = (batch_size x seq_len x hidden_size)

        if self.featrep == 'zeropad':
            lstm_out = self.last_timestep(lstm_out, lengths)  
        else:
            lstm_out = lstm_out[:,-1,:]

        fc_outputs = self.fc(lstm_out)      
        last_outputs = self.softmax(fc_outputs)
        return last_outputs

    def last_timestep(self, outputs, lengths):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """            
        
        if self.bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths, self.use_cnn)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths, self.use_cnn)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths, use_cnn):
        if use_cnn:  # If we've used use_cnn cnn
            return outputs[:, -1, :]
        else:
            # Index of the last output for each sequence.
            idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                                   outputs.size(2)).unsqueeze(1)
            return outputs.gather(1, idx).squeeze()        
