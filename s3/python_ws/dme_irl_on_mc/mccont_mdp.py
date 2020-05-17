"""
  @author: yigit.yildirim@boun.edu.tr
"""

import os.path
import numpy as np
import gym
from tqdm import tqdm


class MCContMDP:
    def __init__(self):
        self.x_div = 60
        self.v_div = 40

        self.shape = (self.x_div, self.v_div)

        self.num_states = self.x_div * self.v_div
        self.num_actions = 3

        self.data_path = "data/mccont/"
        self.env_path = self.data_path + "env/s:" + \
            str(self.x_div) + "-" + str(self.v_div) + "-a:" + str(self.num_actions) + "/"

        self.is_generated = os.path.isfile(self.env_path + "actions.npy")
        self.is_forward_transition_generated = os.path.isfile(self.env_path + "forward_transitions.npy")
        self.states, self.actions, self.transitions = None, None, None
        self.forward_transitions, self.backward_transitions = [], []

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
        # ======================
        if self.is_forward_transition_generated:
            self.forward_transitions = self.load_np_file(self.env_path + "forward_transitions.npy")
            self.backward_transitions = self.load_np_file(self.env_path + "backward_transitions.npy")
        else:
            self.create_forward_backward_transitions()
            self.is_forward_transition_generated = True

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
        for i, s in enumerate(tqdm(self.states)):
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

    def create_forward_backward_transitions(self):
        # Upon Ersin's 3rd suggestion in issue #39, changing transition representation.
        # F and B transitions will be lookup tables, such as f[x]=[[49, 50], [50]]. (a=0 leads to (49 and 50) and so on)
        # For each state, we keep <num_actions> lists. Each list keeps possible destinations
        print('Preparing forward and backward transitions:')
        for s in tqdm(range(self.num_states)):
            f = []
            b = []
            for a in range(self.num_actions):
                f.append(self.transitions[s, a].nonzero()[0])
                start_states = self.transitions[:, a, s].nonzero()[0]  # for backward
                for start_state in start_states:
                    b.append((start_state, a))
            self.forward_transitions.append(f)
            self.backward_transitions.append(b)

        self.save_np_file(self.env_path + "forward_transitions.npy", np.array(self.forward_transitions))
        self.save_np_file(self.env_path + "backward_transitions.npy", np.array(self.backward_transitions))

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
            # print('get_start_state+')
            s = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
            start_candidates = self.find_closest_states(s)

            found_start = False
            for start_candidate in start_candidates:
                if np.any(self.transitions[start_candidate, :, start_candidate] < 1):
                    self.start_state_id = start_candidate
                    found_start = True
                    break

            if not found_start:  # start candidates are trapping
                return self.get_start_state()

        # print('get_start_state-')
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
