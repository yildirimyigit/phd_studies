"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import torch.nn as nn
import time

dev = torch.device("cuda")
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=dev)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], device=dev)

model = nn.Linear(1, 1).cuda()

loss = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.1)
t0 = time.time()
for i in range(500):
    s = model(x)
    l = loss(s, y)
    l.backward()
    optim.step()
    optim.zero_grad()

    [w, b] = model.parameters()
    print(w[0][0].item(), b.item())
t1 = time.time()

print(t1-t0)
