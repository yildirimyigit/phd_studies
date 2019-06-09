"""
  @author: yigit.yildirim@boun.edu.tr
  @author: irmak.guzey@boun.edu.tr
"""
import numpy as np
from utils import *


class IRLMDP:
    def __init__(self, path='data/'):
        self.path = path
        self.states, self.actions, self.transition = self.create_env()
        self.state_list = self.get_state_list()
        self.gamma = 0.9
        # TODO: start and goal states
        self.start_id = self.get_start_state()
        self.goal_id = self.get_goal_state()

    # returns previously generated states and actions
    def create_env(self):
        return np.load(self.path + 'states.npy'), np.load(self.path + 'actions.npy'), \
               np.load(self.path + 'transitions.npy')

    def find_closest_state(self, state):
        min_ind = -1
        min_dist = np.inf
        for i in range(len(self.states)):
            dist = state.distance(self.states[i])
            if dist < min_dist:
                min_dist = dist
                min_ind = i

        return min_ind

    # Methods specific to MC.Cont. environment
    def get_start_state(self):
        return self.find_closest_state(State(0, 0))

    def get_goal_state(self):
        return self.find_closest_state(State(0.45, 0.055))

    def get_state_list(self):
        states = []
        for s in self.states:
            states.append([s.x, s.v])

# Given state is goal or not
# def is_goal(state):
#     return state.dg == 0
