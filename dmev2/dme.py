"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
from irl_agent import IRLAgent


class DME:
    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.iter_count = 50000

        if self.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        self.error = torch.zeros(self.iter_count, device=dev)
        self.agent = IRLAgent(dev)
