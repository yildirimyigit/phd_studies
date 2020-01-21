"""
  @author: yigit.yildirim@boun.edu.tr
"""
import torch
import os
import gym
import numpy as np
from tqdm import tqdm


class MCContMDP:
    def __init__(self):
        self.x_div = 60
        self.y_div = 40
        self.num_actions = 10

        self.data_path = "data/mccont/"
        self.model_path = self.data_path + "model/" + \
            str(self.x_div) + "-" + str(self.y_div) + "-" + str(self.num_actions) + "/"

        self.shape = (self.x_div, self.y_div)
        self.num_states = self.x_div * self.y_div

        self.is_generated_before = os.path.isfile(self.model_path + "actions.pt")
        self.states, self.actions, self.transitions = None, None, None

        self.generate_environment()

    def generate_environment(self):
        if self.is_generated_before:
            self.states = self.load_tensor("states.pt")
            self.actions = self.load_tensor("actions.pt")
            self.transitions = self.load_tensor("transitions.pt")
        else:
            self.create_data()
            self.is_generated_before = True

    def create_data(self):
        try:
            os.makedirs(self.model_path)
        except:
            pass
        env = gym.make('MountainCarContinuous-v0')

        states = np.zeros((self.num_states, 2))
        actions = np.zeros((self.num_actions, 1))
        transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

        min_act = env.unwrapped.min_action
        max_act = env.unwrapped.max_action
        min_x = env.unwrapped.min_position
        max_x = env.unwrapped.max_position
        max_v = env.unwrapped.max_speed
        min_v = -max_v

        act_step = (max_act - min_act) / self.num_actions
        x_step = (max_x - min_x) / self.x_div
        v_step = (max_v - min_v) / self.y_div

        # creating self.states
        x_cur = min_x + x_step / 2
        i = 0
        while x_cur < max_x:
            j = 0
            v_cur = min_v + v_step / 2
            while v_cur < max_v:
                states[i * self.y_div + j] = [x_cur, v_cur]
                v_cur += v_step
                j += 1
            x_cur += x_step
            i += 1

        self.states = torch.from_numpy(states)

        # creating self.actions
        cur_act = min_act + act_step / 2
        for i in range(self.num_actions):
            actions[i] = [cur_act]
            cur_act += act_step

        self.actions = torch.from_numpy(actions)

        env.reset()
        for i in tqdm(range(len(states))):
            s = states[i]
            for j, a in enumerate(actions):
                env.unwrapped.state = np.array(s)
                next_s, _, _, _ = env.step(a)
                k = self.find_closest_state(next_s)
                transitions[i, j, k] = 1

        env.close()
        self.transitions = torch.from_numpy(transitions)

        self.save_tensor(self.states, "states.pt")
        self.save_tensor(self.actions, "actions.pt")
        self.save_tensor(self.transitions, "transitions.pt")

    def find_closest_state(self, state):
        min_ind = -1
        min_dist = np.inf
        for i, s in enumerate(self.states):
            dist = (s[0].item() - state[0]) ** 2 + (s[1].item() - state[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                min_ind = i

        return min_ind

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
        closest = self.find_closest_state(s)

        # closest = 604
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
