"""
  @author: yigit.yildirim@boun.edu.tr
"""

import os.path
import numpy as np
import gym


class MCContMDP:
    def __init__(self):
        self.x_div = 100
        self.v_div = 100

        self.shape = (self.x_div, self.v_div)

        self.num_states = self.x_div * self.v_div
        self.num_actions = 2

        self.data_path = "data/mccont/"
        self.env_path = self.data_path + "env/s:" + \
            str(self.x_div) + "-" + str(self.v_div) + "-a:" + str(self.num_actions) + "/"

        self.is_generated = os.path.isfile(self.env_path + "actions.npy")
        self.states, self.actions, self.transitions = None, None, None

        self.generate_environment()

        self.start_state_id = None
        self.get_start_state()

    def generate_environment(self):
        if self.is_generated:
            self.states = self.load_np_file(self.env_path + "states.npy")
            self.actions = self.load_np_file(self.env_path + "actions.npy")
            self.transitions = self.load_np_file(self.env_path + "transitions.npy")
        else:
            self.create_data()
            self.is_generated = True

    def create_data(self):
        try:
            os.makedirs(self.env_path)
        except OSError:
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

        x_vals = np.linspace(min_x+x_step/2, max_x-x_step/2, self.x_div)
        v_vals = np.linspace(min_v+v_step/2, max_v-v_step/2, self.v_div)

        for i, x in enumerate(x_vals):
            for j, v in enumerate(v_vals):
                self.states[i * self.v_div + j] = [x, v]

        # self.actions = np.reshape(np.linspace(min_act+act_step/2, max_act-act_step/2, self.num_actions),
        #                           (self.num_actions, 1))
        # Upon Ersin's suggestion, now A = {-1, 1}
        self.actions = np.reshape(np.linspace(min_act, max_act, self.num_actions), (self.num_actions, 1))

        env.reset()
        for i, s in enumerate(self.states):
            for j, a in enumerate(self.actions):
                env.unwrapped.state = np.array(s)
                next_s, _, _, _ = env.step(a)
                closest_states = self.find_closest_states(next_s)
                prob_per_state = 1/len(closest_states)
                for k in closest_states:
                    self.transitions[i, j, k] = prob_per_state

        env.close()

        self.save_np_file(self.env_path + "states.npy", self.states)
        self.save_np_file(self.env_path + "actions.npy", self.actions)
        self.save_np_file(self.env_path + "transitions.npy", self.transitions)

    def save_np_file(self, filepath, m_array):
        np.save(filepath, m_array)

    def load_np_file(self, filepath):
        return np.load(filepath, allow_pickle=True, encoding='latin1')

    def find_closest_states(self, state):
        indices = []
        min_dist = np.inf
        for i, s in enumerate(self.states):
            dist = np.round((s[0]-state[0])**2 + (s[1]-state[1])**2, 12)  # rounding to 12 decimals
            # because of a float calc. error in Python.
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
            dist = (action-self.actions[i])**2
            if dist < min_dist:
                min_dist = dist
                min_ind = i
        return min_ind

    def get_start_state(self):
        if self.start_state_id is None:
            print('get_start_state+')
            s = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
            self.start_state_id = np.random.choice(self.find_closest_states(s), 1)[0]

            # redo if trapping state, bad implementation here
            while np.all(self.transitions[self.start_state_id, :, self.start_state_id] == 1):
                print('++searching_nontrapping_start_state')
                s = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
                self.start_state_id = self.find_closest_states(s)[0]

        print('get_start_state-')
        return np.array(self.start_state_id)

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
