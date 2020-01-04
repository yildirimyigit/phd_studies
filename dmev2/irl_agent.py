"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch


class IRLAgent:
    def __init__(self):
        self.env = IRLMDP()
        self.exp_fc = torch.zeros()