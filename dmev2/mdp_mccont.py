"""
  @author: yigit.yildirim@boun.edu.tr
"""
import torch
import os
import gym


class MCContMDP:
    def __init__(self):
        self.data_path = "data/mccont/"

        self.x_div = 60
        self.y_div = 40
        self.num_actions = 10

        self.shape = (self.x_div, self.y_div)
        self.num_states = self.x_div * self.y_div

        self.is_generated_before = os.path.isfile(self.data_path + "actions.pt")
        self.states, self.actions, self.transitions = None, None, None

        self.generate_environment()

    def generate_environment(self):
        if self.is_generated_before:
            self.states = self.load_tensor(self.data_path + "states.pt")
            self.actions = self.load_tensor(self.data_path + "actions.pt")
            self.transitions = self.load_tensor(self.data_path + "transitions.pt")
        else:
            self.create_data()
            self.is_generated_before = True

    def create_data(self):
        pass

    def find_closest_state(self, state):
        pass

    def find_closest_action(self, state):
        pass

    def get_start_state(self):
        s = torch.array([torch.rand() * 0.2 - 0.6, 0])
        closest = self.find_closest_state(s)

        # closest = 604
        return torch.tensor(closest)

    def get_goal_state(self):
        goal = []
        for i, s in enumerate(self.states):
            if s[0] > 0.45:  # 0.45 is env.goal_position
                goal.append(i)
        return torch.tensor(goal)

    def load_tensor(self, path):
        pass

    def save_tensor(self, path):
        pass
