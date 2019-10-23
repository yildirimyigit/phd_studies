"""
  @author: yigit.yildirim@boun.edu.tr
"""
import numpy as np
from utils import *

import gym
from gym import wrappers

vid_ep = 20


class Environment(object):
    def __init__(self, state_div=40, action_div=10):
        self.state_div = state_div
        self.action_div = action_div
        self.action_space = (-1, 1)
        self.state_space = {'x': (-1.2, 0.6), 'v': (-0.07, 0.07)}
        self.state_list = []    # list of State objects
        self.action_list = []   # list of Action objects

    def initialize_actions(self):
        print('+ Environment.initialize_actions()')
        action_interval = self.action_space[1]-self.action_space[0]
        change = action_interval / self.action_div
        for i in range(self.action_div):
            self.action_list.append(Action(self.action_space[0] + (0.5 + i) * change))

    def initialize_states(self):
        print('+ Environment.initialize_states()')
        x_interval = self.state_space['x'][1] - self.state_space['x'][0]
        v_interval = self.state_space['v'][1] - self.state_space['v'][0]
        x_change = x_interval / self.state_div
        v_change = v_interval / self.state_div
        for i in range(self.state_div):
            x = self.state_space['x'][0] + (0.5 + i) * x_change
            for j in range(self.state_div):
                v = self.state_space['v'][0] + (0.5 + j) * v_change
                self.state_list.append(State(x, v))

    # def initialize_transitions(self):
    #     print('+ Environment.initialize_states()')
    #     x_interval = self.state_space['x'][1] - self.state_space['x'][0]
    #     v_interval = self.state_space['v'][1] - self.state_space['v'][0]
    #     x_change = x_interval / self.state_div
    #     v_change = v_interval / self.state_div
    #     for i in range(self.state_div):
    #         x = self.state_space['x'][0] + (0.5 + i) * x_change
    #         for j in range(self.state_div):
    #             v = self.state_space['v'][0] + (0.5 + j) * v_change
    #             self.state_list.append(State(x, v))

    def transition(self, env, state, action):
        env.unwrapped.state = [state.x, state.v]
        next_state, _, _, _ = env.step([action.force])

        state_prob_dist = np.zeros(len(self.state_list))

        # This part is specific to the deterministic environments
        closest = self.find_closest_state(State(next_state[0], next_state[1]))
        state_prob_dist[closest] = 1
        return state_prob_dist

    def find_closest_state(self, state):
        min_ind = -1
        min_dist = np.inf
        for i in range(len(self.state_list)):
            dist = state.distance(self.state_list[i])
            if dist < min_dist:
                min_dist = dist
                min_ind = i

        return min_ind

    def find_closest_action(self, action):
        min_ind = -1
        min_dist = np.inf
        for i in range(len(self.action_list)):
            dist = action.distance(self.action_list[i])
            if dist < min_dist:
                min_dist = dist
                min_ind = i
        return min_ind

    def save_states(self, file_name):
        print('+ Environment.save_states()')
        np.save(file_name, np.asarray(self.state_list))

    def save_actions(self, file_name):
        print('+ Environment.save_actions()')
        np.save(file_name, np.asarray(self.action_list))

    def save_transitions(self, file_name):
        print('+ Environment.save_transitions()')
        nof_states = len(self.state_list)
        transition_mat = np.zeros([nof_states, len(self.action_list), nof_states], dtype=float)  # T[s][a][s']

        env = gym.make('MountainCarContinuous-v0')
        # env = wrappers.Monitor(env, './tmp/', force=True, video_callable=lambda episode_id: episode_id % vid_ep == 0)
        env.reset()

        for i in range(nof_states):
            for j in range(len(self.action_list)):
                transition_mat[i, j, :] = self.transition(env, self.state_list[i], self.action_list[j])
                env.reset()

        env.close()
        np.save(file_name, transition_mat)
        print('- Environment.save_transitions()')

    def initialize_environment(self):
        print('+ Environment.initialize_environment()')
        self.initialize_states()
        self.initialize_actions()


def print_state(s):
        print('x: {0}, v: {1}'.format(s.x, s.v))


def print_action(a):
    print('force: {0}'.format(a.force))
