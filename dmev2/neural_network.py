"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, nn_arch):
        super().__init__()
        self.layers = []

        num_layers = 0
        while num_layers < len(nn_arch)-1:   # nn_arch: a list of numbers representing the neurons per layer
            self.layers.append(nn.Linear(nn_arch[num_layers], nn_arch[num_layers+1]))
            num_layers += 1

    def forward(self, x):
        # All layer activations are sigmoid except for the last one

        for i, layer in enumerate(self.layers):
            if i != len(self.layers)-1:
                x = torch.sigmoid(layer(x))
            else:
                x = layer(x)

        return x


data = torch.rand([9], dtype=torch.float32)
print(data)
net = Net([3, 32, 32, 1])
print(net(data.view(-1, 3)))
