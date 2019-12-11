"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import torch.nn as nn
import torch.nn.functional as funct


class NN(nn.Module):
    def __init__(self, nn_arch=None):
        super().__init__()

        if nn_arch is None:
            nn_arch = [2, 32, 32, 1]

        self.fc1 = nn.Linear(nn_arch[0], nn_arch[1])
        self.fc2 = nn.Linear(nn_arch[1], nn_arch[2])
        self.fc3 = nn.Linear(nn_arch[2], nn_arch[3])

    def forward(self, x):
        x = funct.relu(self.fc1(x))
        x = funct.relu(self.fc2(x))
        x = funct.tanh(self.fc3(x))

        return x


net = NN([2, 16, 32, 1])
X = torch.rand((2, 1))
X = X.view(-1, 2)

out = net(X)
print(out)
