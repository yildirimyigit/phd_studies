"""
  @author: yigit.yildirim@boun.edu.tr
"""

from utils import *
import numpy as np


# Loads previously recorded demonstrations to create trajectories
def run(path='data/'):
    demonstrations = []
    for i in range(9):
        demonstrations.append(np.load(path+'big_t'+str(i)+'.npy'))



















if __name__ == "__main__":
	run()
