import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class bLSTM(nn.Module):
    def __init__(self, config):
        super(bLSTM, self).__init__()
        
        # intilize structure of the lstm
        self.lstm = nn.LSTMCell(config['input_dim'],config['hidden_dim'])
        self.hidden2char = nn.Linear(config['hidden_dim'],config['output_dim'])
        
        # zero intialize the hidden vectors and cell state( no info at start ) 
        self.initialCx = torch.zeros(config['batch_size'],config['hidden_dim']).cuda()
        self.initialHx = torch.zeros(config['batch_size'],config['hidden_dim']),cuda()
        
        self.softmax = nn.LogSoftMax(dim = 1) # dim 1 for sequence length (1 character)
       
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        cx = self.initalCx
        hx = self.initalHx
        out = []
        
        for i in range(sequence.shape[1]):
            hx, cx = self.lstm(sequence[:,i,:],(hx,cx))
            out.append(self.softmax(self.hidden2char(hx)))
        
        out = torch.stack(out, 1)
        
        return out


class bGRU(nn.Module):
    def __init__(self, config):
        super(bGRU, self).__init__()
        
        # intilize structure of the lstm
        self.gru = nn.GRUCell(config['input_dim'],config['hidden_dim'])
        self.hidden2char = nn.Linear(config['hidden_dim'],config['output_dim'])
        
        # zero intialize the hidden vectors and cell state( no info at start ) 
        self.initialHx = torch.zeros(config['batch_size'],config['hidden_dim']),cuda()
        
        self.softmax = nn.LogSoftMax(dim = 1) # dim 1 for sequence length (1 character)
       
        
        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        
    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        hx = self.initalHx
        out = []
        
        for i in range(sequence.shape[1]):
            hx = self.lstm(sequence[:,i,:],hx)
            out.append(self.softmax(self.hidden2char(hx)))
        
        out = torch.stack(out, 1)
        
        return out