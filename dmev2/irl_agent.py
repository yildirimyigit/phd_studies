"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
from mdp_mccont import MCContMDP


class IRLAgent:
    def __init__(self):
        self.env = MCContMDP()
        self.exp_fc = torch.zeros()