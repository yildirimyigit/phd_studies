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
        self.start_id = self.get_start_state()
        self.goal_id = self.get_goal_state()

    # returns previously generated states, actions and transitions
    def create_env(self):
        return np.load(self.path + 'states.npy', allow_pickle=True), \
               np.load(self.path + 'actions.npy', allow_pickle=True), \
               np.load(self.path + 'transitions.npy', allow_pickle=True)

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
        return states

# Given state is goal or not
# def is_goal(state):
#     return state.dg == 0


class Objectworld:
    def __init__(self, dim=8, stochasticity=0.0, nof_colors=10):  # TODO: stochasticity is 0.3 by definition
        self.dim = dim
        self.stochasticity = stochasticity
        self.nof_actions = 5    # by definition
        self.gamma = 0.9
        self.nof_colors = nof_colors

        self.states = self.actions = self.transition = np.empty()
        self.state_list = []

        self.create_env()
        self.start_id = self.get_start_state()
        self.goal_id = self.get_goal_state()

    # returns previously generated states, actions and transitions
    def create_env(self):
        state_list = []
        action_list = []
        transition_matrix = []

        for i in range(self.dim):
            for j in range(self.dim):
                state_list.append(ObjectworldState(i, j))
                self.state_list.append([i, j])

        action_list.append(ObjectworldAction(0, -1))
        action_list.append(ObjectworldAction(1, 0))
        action_list.append(ObjectworldAction(0, 1))
        action_list.append(ObjectworldAction(0, -1))
        action_list.append(ObjectworldAction(0, 0))



        self.states = np.asarray(state_list)
        self.actions = np.asarray(action_list)
        self.transition = np.asarray(transition_matrix)

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

    class Object:
        def __init__(self, location=0, color=0):
            self.location = location
            self.color = color
