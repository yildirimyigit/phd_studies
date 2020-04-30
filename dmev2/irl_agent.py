"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
import torch.optim as optim
from mdp_mccont import MCContMDP
from neural_network import Net


class IRLAgent:
    def __init__(self, device):
        self.env = MCContMDP()
        self.features = self.calculate_features().to(device)
        # self.features.requires_grad_(True)  # Because it is the input to the network
        self.emp_fc = torch.zeros(device=device)
        self.exp_fc = torch.zeros(device=device)

        self.calculate_emp_fc()

        self.model = Net([3, 32, 32, 1])

        self.lr = 1e-3
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def calculate_emp_fc(self):
        pass

    def calculate_rewards(self):
        self.model.zero_grad()

        self.model()

    # This part must be specific to the environment. I'm just normalizing the states in case of MCCont and Gridworld
    # environments since they are simple enough.
    def calculate_features(self):
        min_vals, _ = self.env.states.min(dim=0)  # min_vals of each dimension
        max_vals, _ = self.env.states.max(dim=0)
        diff = max_vals - min_vals

        normalized = self.env.states - min_vals
        normalized /= diff

        return normalized

    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

