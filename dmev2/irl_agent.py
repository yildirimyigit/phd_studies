"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
from mdp_mccont import MCContMDP


class IRLAgent:
    def __init__(self, device):
        self.env = MCContMDP()
        self.emp_fc = torch.zeros(device=device)
        self.exp_fc = torch.zeros(device=device)

        self.calculate_emp_fc()

    def calculate_emp_fc(self):
        pass

    def calculate_rewards(self):
        pass

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

