"""
  @author: yigit.yildirim@boun.edu.tr
"""

from utils import *
import numpy as np
from environment import Environment


def generate_model():
    env = Environment()
    env.initialize_environment()

    env.save_states('data/states.npy')
    env.save_actions('data/actions.npy')
    env.save_transitions('data/transitions.npy')


# Loads previously recorded demonstrations to create trajectories
def generate_trajectories(path='data/'):
    demonstrations = []
    for i in range(9):
        demonstrations.append(np.load(path+'big_t'+str(i)+'.npy'))

    # TODO compute states of trajectories and save them in trajectories.npy


if __name__ == "__main__":
    generate_model()
    # generate_trajectories()
