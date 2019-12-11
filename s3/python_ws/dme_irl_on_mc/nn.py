"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim


class NN(nn.Module):
    def __init__(self, nn_arch=None, lr=1e-3):
        super().__init__()

        if nn_arch is None:
            nn_arch = [2, 32, 32, 1]
        self.lr = lr

        self.fc1 = nn.Linear(nn_arch[0], nn_arch[1])
        self.fc2 = nn.Linear(nn_arch[1], nn_arch[2])
        self.fc3 = nn.Linear(nn_arch[2], nn_arch[3])

    def forward(self, x):
        x = funct.relu(self.fc1(x))
        x = funct.relu(self.fc2(x))
        x = funct.tanh(self.fc3(x))

        return x


def setdata():
    xten = torch.linspace(0, 1, 1024)
    x = torch.split(xten, 32, dim=0)
    y = torch.split((0.2 + 0.4 * xten ** 2 + 0.3 * xten * torch.sin(15 * xten) + 0.05 * torch.cos(50 * xten)), 32,
                    dim=0)
    # x = torch.rand((2, 2))
    return x, y


if __name__ == "__main__":
    net = NN([2, 16, 16, 1])

    data, label = setdata()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    EPOCHS = 1000
    for epoch in range(EPOCHS):
        for dat, lab in zip(data, label):   # data is a pair that contains batches of (X, y)
            # print(lab)
            net.zero_grad()
            out = net(dat.view(-1, 2))
            loss = funct.mse_loss(out, lab)
            loss.backward()
            optimizer.step()
        print(loss)


# print(x)
# out = net(X)
# print(out)
