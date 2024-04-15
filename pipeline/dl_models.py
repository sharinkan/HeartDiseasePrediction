import torch
import torch.nn as nn
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
        self.fc_layers = [nn.Linear(prev_lay, cur_lay) for prev_lay, cur_lay in pairwise(struct)]
        self.activations = activation_layers
        
        if isinstance(self.activations, list):
            self.activations = [self.activations()]
        else:
            self.activations = [self.activations()] * (len(self.fc_layers) - 1) # one activ after each layer
            
        assert len(self.fc_layers) > len(self.activations), "more activation than fc layers"
        
            
    def forward(self, x):
        for layer, acti in zip_longest(self.fc_layers, self.activations, fillvalue=lambda _x : _x):
            x = acti(layer(x))
        return x
        

if __name__ == "__main__":
    input_dim = [10,32,64,2]
    model = MLP(input_dim, nn.ReLU)

    input_data = torch.randn(32, input_dim[0])

    # Forward pass
    output = model(input_data)
    print("Output shape:", output)