import torch
import torch.nn as nn
from typing import List, Union, Callable
from itertools import zip_longest

import numpy as np

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
        self.fc_layers = nn.ModuleList([nn.Linear(prev_lay, cur_lay) for prev_lay, cur_lay in pairwise(struct)])
        self.activations = activation_layers
        
        if isinstance(self.activations, list):
            self.activations = nn.ModuleList([act() for act in self.activations])
        else:
            self.activations = nn.ModuleList([self.activations()] * (len(self.fc_layers) - 1))  # one activ after each layer
            
        assert len(self.fc_layers) > len(self.activations), "more activation than fc layers"
        
            
    def forward(self, x):
        for layer, acti in zip_longest(self.fc_layers, self.activations, fillvalue=None):
            x = layer(x)
            if acti is None:
                continue
            
            x = acti(x)
        return x
    
    
class CombinedMLP(nn.Module):
    def __init__(self, MLPS : List[MLP]):
        super(CombinedMLP, self).__init__()
        self.MLPS = MLPS # each output shape is 1
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
    
    def parameters(self, *args, **kwargs):
        return list(super(CombinedMLP, self).parameters(*args, **kwargs)) + [param for mlp in self.MLPS for param in mlp.parameters()]
    
    def train(self, *args, **kwargs):
        super(CombinedMLP, self).train(*args, **kwargs)
        for sub_model in self.MLPS:
            sub_model.train(*args, **kwargs)
            
    def eval(self, *args, **kwargs):
        super(CombinedMLP, self).eval(*args, **kwargs)
        for sub_model in self.MLPS:
            sub_model.eval(*args, **kwargs)


    def to(self, *args, **kwargs):
        super(CombinedMLP, self).to(*args, **kwargs)
        for sub_model in self.MLPS:
            sub_model.to(*args, **kwargs)
        
    
        
if __name__ == "__main__":
    
    x = torch.randn(2, 1, requires_grad=True)
    print(x.grad)
    # Perform some operations
    y = x * 2
    z = y.mean()

    # Backpropagate through z
    # z.backward()
    ff = torch.cat([y,y],dim=1)
    print(y.shape)
    print(ff.shape)
    ll = ff.mean()
    ll.backward()

    # Check gradient history of x
    print(x.grad)