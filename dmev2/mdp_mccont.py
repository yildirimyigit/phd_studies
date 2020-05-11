"""
  @author: yigit.yildirim@boun.edu.tr
"""
import torch
import os
import gym
import numpy as np
from tqdm import tqdm


class MCContMDP:
    def __init__(self, state_space=(120, 80)):
        self.state_space = state_space
        self.num_actions = 10

        self.data_path = "data/mccont/"
        self.model_path = self.data_path + "model/" + "s:"
        for i in self.state_space:
            self.model_path += str(i) + "-"
        self.model_path += "a:" + str(self.num_actions) + "/"

        self.num_states = int(np.prod(self.state_space))

        self.states, self.actions, self.transitions = None, None, None
        self.generate_environment(os.path.isfile(self.model_path + "actions.pt"))  # if exists, don't generate again

    def generate_environment(self, generated_before):
        if generated_before:
            self.states = self.load_tensor("states.pt")
            self.actions = self.load_tensor("actions.pt")
            self.transitions = self.load_tensor("transitions.pt")
        else:
            self.create_data()

    # this method is specific to my definition of the environment. May be improved to be more generic
    def create_data(self):
        try:
            os.makedirs(self.model_path)
        except OSError:
            pass
        env = gym.make('MountainCarContinuous-v0')

        self.states = torch.zeros(self.num_states, len(self.state_space))
        self.actions = torch.zeros(self.num_actions, 1)
        self.transitions = torch.zeros((self.num_states, self.num_actions, self.num_states))

        min_act = env.unwrapped.min_action
        max_act = env.unwrapped.max_action
        min_x = env.unwrapped.min_position
        max_x = env.unwrapped.max_position
        max_v = env.unwrapped.max_speed
        min_v = -max_v

        act_step = (max_act - min_act) / self.num_actions
        x_step = (max_x - min_x) / self.state_space[0]
        v_step = (max_v - min_v) / self.state_space[1]

        x_vals = torch.linspace(min_x + x_step / 2, max_x - x_step / 2, self.state_space[0])
        v_vals = torch.linspace(min_v + v_step / 2, max_v - v_step / 2, self.state_space[1])

        for i, x in enumerate(x_vals):
            for j, v in enumerate(v_vals):
                self.states[i * self.state_space[1] + j] = torch.Tensor([x, v])

        self.actions = torch.linspace(min_act + act_step / 2, max_act - act_step / 2, self.num_actions)
        self.actions = self.actions.view(self.num_actions, 1)

        env.reset()
        np_states = self.states.numpy()
        for i in tqdm(range(self.num_states)):
            s = self.states[i]
            for j, a in enumerate(self.actions):
                env.unwrapped.state = s.numpy()
                next_s, _, _, _ = env.step(a.numpy())
                closest_states = self.find_closest_states(next_s, np_states)
                prob_per_state = 1 / len(closest_states)
                for k in closest_states:
                    self.transitions[i, j, k] = prob_per_state

        env.close()

        self.save_tensor(self.states, "states.pt")
        self.save_tensor(self.actions, "actions.pt")
        self.save_tensor(self.transitions, "transitions.pt")

    def find_closest_states(self, state, np_states):
        indices = []
        min_dist = np.inf
        for i, s in enumerate(np_states):
            dist = (s[0] - state[0]) ** 2 + (s[1] - state[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                indices.clear()
                indices.append(i)
            elif dist == min_dist:
                indices.append(i)
        return indices

    def find_closest_action(self, action):
        min_ind = -1
        min_dist = np.inf
        for i in range(len(self.actions)):
            dist = (action - self.actions[i].item()) ** 2
            if dist < min_dist:
                min_dist = dist
                min_ind = i
        return min_ind

    def get_start_state(self):
        s = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        # choose one uniformly among all closest candidates
        closest = np.random.choice(self.find_closest_states(s, self.states.numpy()), 1)

        return torch.tensor(closest)

    def get_goal_state(self):
        goal = []
        for i, s in enumerate(self.states):
            if s[0] > 0.45:  # 0.45 is env.goal_position
                goal.append(i)
        return torch.tensor(goal)

    def load_tensor(self, name):
        return torch.load(self.model_path + name)

    def save_tensor(self, tnsr, name):
        torch.save(tnsr, self.model_path + name)


if __name__ == "__main__":
    mcc = MCContMDP()
    print("created")
