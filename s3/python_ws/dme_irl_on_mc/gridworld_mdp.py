"""
  @author: yigit.yildirim@boun.edu.tr
"""

import os
import numpy as np


class GridworldMDP:
    def __init__(self):
        self.data_path = "data/gridworld/"
        self.x_div = 4
        self.y_div = 4

        self.shape = (self.x_div, self.y_div)

        self.num_states = self.x_div * self.y_div
        self.num_actions = 4

        self.is_generated = os.path.isfile(self.data_path + "actions.npy")
        self.states, self.actions, self.transitions = None, None, None

        self.generate_environment()

        self.start_state_id = None
        self.get_start_state()

    def generate_environment(self):
        if self.is_generated:
            self.states = self.load_np_file(self.data_path + "states.npy")
            self.actions = self.load_np_file(self.data_path + "actions.npy")
            self.transitions = self.load_np_file(self.data_path + "transitions.npy")
        else:
            self.create_data()
            self.is_generated = True

    def create_data(self):
        try:
            os.makedirs(self.data_path)
        except:
            pass

        self.states = np.zeros((self.num_states, 2))
        self.actions = np.zeros((self.num_actions, 1))
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

        # states
        for i in range(self.x_div):
            for j in range(self.y_div):
                self.states[i * self.x_div + j] = [i, j]

        # actions
        for i in range(4):
            self.actions[i] = i

        # transitions
        for i in range(self.num_states):
            for j in range(self.num_actions):
                k = i
                if j == 0:
                    if i >= self.x_div:  # moving up in the first line doesn't have affect the position
                        k = i - self.x_div
                elif j == 1:
                    if (i+1) % self.x_div != 0:
                        k = i + 1
                elif j == 2:
                    if i + self.x_div < self.num_states:
                        k = i + self.x_div
                elif j == 3:
                    if i % self.x_div != 0:
                        k = i - 1

                self.transitions[i, j, k] = 1

        self.save_np_file(self.data_path + "states.npy", self.states)
        self.save_np_file(self.data_path + "actions.npy", self.actions)
        self.save_np_file(self.data_path + "transitions.npy", self.transitions)

    def get_start_state(self):
        self.start_state_id = 0
        return np.array(0)

    def get_goal_state(self):
        return np.array(11)

    def save_np_file(self, filepath, m_array):
        np.save(filepath, m_array)

    def load_np_file(self, filepath):
        return np.load(filepath, allow_pickle=True)


if __name__ == "__main__":
    mcc = GridworldMDP()
    print(mcc.get_start_state())
    print(mcc.get_goal_state())
    print("here")
