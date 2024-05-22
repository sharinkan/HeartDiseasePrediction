import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import List, Union, Callable
from itertools import zip_longest


def pairwise(iterable): # [1,2,3,4,5] -> [(1,2), (2,3), (3,4),(4,5)]
    iterator = iter(iterable)
    prev = next(iterator)
    for item in iterator:
        yield (prev, item)
        prev = item
            
class MLP(nn.Module):
    def __init__(self, struct : List[int] = None, activation_layers : Union[List[Callable],  Callable] = nn.ReLU):
        super(MLP, self).__init__()
        # https://stackoverflow.com/questions/62937388/pytorch-dynamic-amount-of-layers

        model_struct_layer = list(pairwise(struct))

        self.fc_layers = nn.ModuleList([nn.Linear(prev_lay, cur_lay) for prev_lay, cur_lay in model_struct_layer])
        self.bns = nn.ModuleList([nn.BatchNorm1d(cur_lay) for _, cur_lay in model_struct_layer[:-1]])
        self.activations = activation_layers
        
        if isinstance(self.activations, list):
            self.activations = nn.ModuleList([act() for act in self.activations])
        else:
            self.activations = nn.ModuleList([self.activations()] * (len(self.fc_layers) - 1))  # one activ after each layer

        
            
        assert len(self.fc_layers) > len(self.activations), "more activation than fc layers"
        
            
    def forward(self, x):
        for layer, acti, bn in zip_longest(self.fc_layers, 
                                                    self.activations, 
                                                    self.bns, 
                                                    fillvalue=None):
            x = layer(x)
            if bn:
                x = bn(x)
            if acti:
                x = acti(x)
        return x
    
    
class CombinedMLP(nn.Module):
    def __init__(self, MLPS : List[MLP]):
        super(CombinedMLP, self).__init__()
        self.MLPS = nn.ModuleList(MLPS) # each output shape is 1
        self.fc = nn.Linear(len(self.MLPS), 1)
        
        
    def forward(self, xs): # x -> 2d array
        combined_x = []
        for x, model in zip(xs, self.MLPS): # assume ordering of feature
            partial_result = model(x.float())
            combined_x.append(partial_result)
        
        combined_x = torch.cat(combined_x, dim=1)
        X = self.fc(combined_x)
        # X = nn.functional.softmax(X,dim=1)
        X = torch.sigmoid(X)
        
        return X # grad is kept so backward will work

# RNNs
# https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification/blob/master/lstm_genre_classifier_pytorch.py
class RNN_BASE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=8, num_layers=2, layer_option : nn.RNNBase = None, stateful : bool = False, hidden_matrices : int = -1):
        super(RNN_BASE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn_layer_option = layer_option(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)


        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim // 2)
        self.linear2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim // 4)
        self.linear3 = nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8)
        self.layer_norm3 = nn.LayerNorm(self.hidden_dim // 8)
        self.linear4 = nn.Linear(self.hidden_dim // 8, output_dim)

        self.hidden = None
        self.stateful = stateful
        self.hidden_matrices = hidden_matrices # RNN only need 1 for hidden, LSTM need 2 -> cell & hidden
        self.batch_size = 0
        self.device = None

        

    def forward(self, input):
        if not(self.batch_size):
            self.batch_size = input.shape[0]

        if self.stateful and hidden:
            if self.hidden_matrices == 1:
                self.hidden._detach()
            else:
                for hidden in self.hidden:
                    hidden._detach()
        else:
            self._init_hidden(self.batch_size, device=input.device)

        out, self.hidden = self.rnn_layer_option(input, self.hidden)
        out = self.layer_norm(out)
        for linear, norm in zip([self.linear1, self.linear2, self.linear3], [self.layer_norm1, self.layer_norm2, self.layer_norm3]):
            out = norm(linear(out))
            
        out = self.linear4(out)
        out = F.sigmoid(out)
        return out
    
    # https://www.kaggle.com/code/purvasingh/text-generation-via-rnn-and-lstms-pytorch#kln-462
    def _init_hidden(self, batch_size, device):
        '''
        Initialize the hidden state of an LSTM/GRU/RNN
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        weights = next(self.parameters()).data
        hidden = (weights.new(self.num_layers, self.hidden_dim).zero_(), ) * self.hidden_matrices
        self.hidden = tuple([ h.to(device) for h in hidden]) if self.hidden_matrices > 1 else hidden[0].to(device)


class RNN(RNN_BASE):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(RNN, self).__init__(input_dim, hidden_dim, output_dim, num_layers, layer_option=nn.RNN, stateful=False, hidden_matrices=1)

class LSTM(RNN_BASE):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTM, self).__init__(input_dim, hidden_dim, output_dim, num_layers, layer_option=nn.LSTM, stateful=False, hidden_matrices=2)
        
if __name__ == "__main__":
    
    mlp1 = MLP([3,1])
    mlp2 = MLP([5,1])
    
    test = CombinedMLP([mlp1, mlp2])
    
    print(list(test.parameters()))
