import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class bLSTM(nn.Module):
    def __init__(self, config):
        super(bLSTM, self).__init__()
        
        # gets the configurations
        self.is_bidirectional = config['bidirectional']
        
        # initializes the structure of the LSTM
        self.lstm = nn.LSTM(config['input_dim'], config['hidden_dim'], config['layers'], 
                            batch_first = True, dropout = config['dropout'], bidirectional = config['bidirectional'])
        
        # initializes the output layer and its activation function
        self.output_layer = nn.Linear(config['hidden_dim'], config['output_dim'])
       
        # creates an initialization for the hidden states and cell states (zeros to denote no info at start) 
        self.initialC = torch.zeros(config['layers'], config['batch_size'], config['hidden_dim'])
        self.initialH = torch.zeros(config['layers'], config['batch_size'], config['hidden_dim'])
        if config['cuda']:
            self.initialC = self.initialC.cuda()
            self.initialH = self.initialH.cuda()
        
        
    def forward(self, sequence, h0 = None, c0 = None):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        # initializes the hidden states and cell states to zeros if they are not given
        if h0 is None:
            h0 = self.initialH
        if c0 is None:
            c0 = self.initialC
        
        # passes the input to the lstm
        hidden_output, (ht, ct) = self.lstm(sequence, (h0, c0))
        
        # passes the output of the hidden layer to the output layer
        output = self.output_layer(hidden_output)
        
        return output, (ht, ct)


class bGRU(nn.Module):
    def __init__(self, config):
        super(bGRU, self).__init__()
        
        # gets the configurations
        self.is_bidirectional = config['bidirectional']
        
        # initializes the structure of the LSTM
        self.gru = nn.GRU(config['input_dim'], config['hidden_dim'], config['layers'], 
                            batch_first = True, dropout = config['dropout'], bidirectional = config['bidirectional'])
        
        # initializes the output layer and its activation function
        self.output_layer = nn.Linear(config['hidden_dim'], config['output_dim'])
       
        # creates an initialization for the hidden states (zeros to denote no info at start)
        self.initialH = torch.zeros(config['layers'], config['batch_size'], config['hidden_dim'], )
        if config['cuda']:
            self.initialH = self.initialH.cuda()
            
            
    def forward(self, sequence, h0 = None):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        # initializes the hidden states to zeros if they are not given
        if h0 is None:
            h0 = self.initialH
            
        # passes the input to the gru
        hidden_output, ht = self.gru(sequence, h0)
        
        # passes the output of the hidden layer to the output layer
        output = self.output_layer(hidden_output)
                
        return output, ht