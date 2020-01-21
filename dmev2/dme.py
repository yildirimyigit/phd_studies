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

        self.error = torch.zeros(self.iter_count, device=dev, requires_grad=True)
        self.agent = IRLAgent(dev)

    def tour(self):
        self.agent.calculate_rewards()
        self.agent.backward_pass()
        self.agent.forward_pass()

        diff = torch.abs(self.agent.emp_fc - self.agent.exp_fc)
        print("Error: ", torch.sum(diff).item)

    def run(self):
        for i in range(self.iter_count):
            self.tour()
