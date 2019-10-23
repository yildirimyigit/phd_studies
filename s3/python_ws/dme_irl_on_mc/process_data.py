"""
  @author: yigit.yildirim@boun.edu.tr
"""

from utils import *
import numpy as np
from environment import Environment


def generate_model(e):
    e.initialize_environment()

    e.save_states('data/mccont_expert_trajs/states.npy')
    e.save_actions('data/mccont_expert_trajs/actions.npy')
    e.save_transitions('data/mccont_expert_trajs/transitions.npy')


# Loads previously recorded demonstrations to create trajectories
# :param e Environment object
def generate_trajectories(e, path='data/mccont_expert_trajs/'):
    trajectories = []
    for i in range(3):
        demonstrations = np.load(path+'t_'+str(i)+'.npy')
        j = 0
        for demonstration in demonstrations:
            print('\rTrajectory file: {0}, Dem: {1} '.format(i, j), end='')
            j += 1
            trajectory = []
            for state_action in demonstration:
                s = e.state_list[e.find_closest_state(State(state_action[0][0], state_action[0][1]))]
                a = e.action_list[e.find_closest_action(Action(state_action[1][0]))]
                trajectory.append([[s.x, s.v], a.force])
            trajectories.append(np.asarray(trajectory))
        print('')
    np.save(path+'trajectories.npy', np.asarray(trajectories))


if __name__ == "__main__":
    env = Environment()
    generate_model(env)
    generate_trajectories(env)
