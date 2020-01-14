"""
  @author: yigit.yildirim@boun.edu.tr
"""

import os.path
import numpy as np
import gym


class MCContMDP:
    def __init__(self):
        self.data_path = "data/mccont/"
        self.x_div = 60
        self.v_div = 30

        self.shape = (self.x_div, self.v_div)

        self.num_states = self.x_div * self.v_div
        self.num_actions = 20

        self.is_generated = os.path.isfile(self.data_path + "actions.npy")
        self.states, self.actions, self.transitions = None, None, None

        self.generate_environment()

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
        env = gym.make('MountainCarContinuous-v0')

        self.states = np.zeros((self.num_states, 2))
        self.actions = np.zeros((self.num_actions, 1))
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))

        min_act = env.unwrapped.min_action
        max_act = env.unwrapped.max_action
        min_x = env.unwrapped.min_position
        max_x = env.unwrapped.max_position
        max_v = env.unwrapped.max_speed
        min_v = -max_v

        act_step = (max_act - min_act) / self.num_actions
        x_step = (max_x - min_x) / self.x_div
        v_step = (max_v - min_v) / self.v_div

        # creating self.states
        x_cur = min_x + x_step / 2
        i = 0
        while x_cur < max_x:
            j = 0
            v_cur = min_v + v_step / 2
            while v_cur < max_v:
                self.states[i * self.v_div + j] = [x_cur, v_cur]
                v_cur += v_step
                j += 1
            x_cur += x_step
            i += 1

        # creating self.actions
        cur_act = min_act + act_step / 2
        for i in range(self.num_actions):
            self.actions[i] = [cur_act]
            cur_act += act_step

        env.reset()
        for i, s in enumerate(self.states):
            for j, a in enumerate(self.actions):
                env.unwrapped.state = np.array(s)
                next_s, _, _, _ = env.step(a)
                k = self.find_closest_state(next_s)
                self.transitions[i, j, k] = 1

        env.close()
        self.save_np_file(self.data_path + "states.npy", self.states)
        self.save_np_file(self.data_path + "actions.npy", self.actions)
        self.save_np_file(self.data_path + "transitions.npy", self.transitions)

    def save_np_file(self, filepath, m_array):
        np.save(filepath, m_array)

    def load_np_file(self, filepath):
        return np.load(filepath, allow_pickle=True)

    def find_closest_state(self, state):
        min_ind = -1
        min_dist = np.inf
        for i, s in enumerate(self.states):
            dist = (s[0]-state[0])**2 + (s[1]-state[1])**2
            if dist < min_dist:
                min_dist = dist
                min_ind = i

        return min_ind

    def find_closest_action(self, action):
        min_ind = -1
        min_dist = np.inf
        for i in range(len(self.actions)):
            dist = (action-self.actions[i])**2
            if dist < min_dist:
                min_dist = dist
                min_ind = i
        return min_ind

    def get_start_state(self):
        s = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        closest = self.find_closest_state(s)

        closest = 604

        return np.array(closest)

    def get_goal_state(self):
        goal = []
        for i, s in enumerate(self.states):
            if s[0] > 0.45:  # 0.45 is env.goal_position
                goal.append(i)
        return np.array(goal)


if __name__ == "__main__":
    mcc = MCContMDP()
#     print(mcc.get_start_state())
#     print(mcc.get_goal_state())
    print(mcc.find_closest_action(np.array([0])))
    print("here")
