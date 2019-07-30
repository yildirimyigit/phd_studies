"""
  @author: yigit.yildirim@boun.edu.tr
  @author: irmak.guzey@boun.edu.tr
"""
import numpy as np
from utils import *
from owutils import *


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
        self.state_list = self.action_list = []

        self.create_env()
        self.start_id = self.get_start_state()
        self.goal_id = self.get_goal_state()

    # returns previously generated states, actions and transitions
    def create_env(self):
        state_list = []
        transition_matrix = []

        for i in range(self.dim):
            for j in range(self.dim):
                state_list.append(ObjectworldState(i, j))
                self.state_list.append([i, j])

        self.action_list.append(ObjectworldAction(-1, 0))
        self.action_list.append(ObjectworldAction(0, 1))
        self.action_list.append(ObjectworldAction(1, 0))
        self.action_list.append(ObjectworldAction(0, -1))
        self.action_list.append(ObjectworldAction(0, 0))

        for s in range(len(self.state_list)):
            for a in range(len(self.action_list)):
                transition_matrix = self.calculate_transition_for_sa(self.state_list[s], self.action_list[a])

        self.states = np.asarray(state_list)
        self.actions = np.asarray(self.action_list)
        self.transition = np.asarray(transition_matrix)

    def calculate_transition_for_sa(self, state, action):
        new_state_probs = np.zeros(len(self.state_list))

        prob_action_realization = 1 - self.stochasticity
        prob_other_action_realization = self.stochasticity / len(self.action_list)

        new_state = [np.clip(state[0] + action.xch, 0, self.dim), np.clip(state[1] + action.ych, 0, self.dim)]
        new_state_probs[new_state[0] * self.dim + new_state[1]] += prob_action_realization  # put prob to intended state

        for other_action in self.action_list:
            if action != other_action:
                new_state = [np.clip(state[0] + other_action.xch, 0, self.dim),
                             np.clip(state[1] + other_action.ych, 0, self.dim)]
                new_state_probs[new_state[0] * self.dim + new_state[1]] += prob_other_action_realization

        return new_state_probs

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
