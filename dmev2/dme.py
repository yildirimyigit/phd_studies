"""
  @author: yigit.yildirim@boun.edu.tr
"""

import torch
from irl_agent import IRLAgent


class DME:
    def __init__(self):
        self.agent = IRLAgent()