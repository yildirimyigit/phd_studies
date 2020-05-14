"""
  @author: yigit.yildirim@boun.edu.tr

  [1]: Kitani 2012, Activity Forecasting
  [2]: Wulfmeier 2016, Maximum Entropy Deep Inverse Reinforcement Learning
"""
import numpy as np
from mccont_mdp import MCContMDP
from neural_network import MyNN, sigm, linear

import sys
import seaborn as sb
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

import gym


class IRLAgent:
    def __init__(self):
        self.env = MCContMDP()
        # initializes nn with random weights
        # self.rew_nn = MyNN(nn_arch=(2, 8, 16, 32, 64, 16, 64, 32, 64, 32, 64, 16, 64, 32, 16, 8, 1),
        #                    acts=[sigm, sigm, sigm, sigm, sigm, sigm, sigm, gaussian, sigm, sigm, sigm,
        #                          sigm, sigm, sigm, sigm, linear])
        self.rew_nn = MyNN(nn_arch=(2, 32, 256, 32, 1), acts=[sigm, sigm, sigm, linear])
        self.state_rewards = np.empty(len(self.env.states), dtype=float)

        self.initialize_rewards()

        # To output the results, the following are used
        self.path = 'data/mccont/'
        self.output_directory_suffix = str(int(time.time()))
        # Creating the output directory for the individual run
        self.output_directory_path = self.path + 'output/' + self.output_directory_suffix + "/"
        self.videodir = self.output_directory_path + "video/"
        self.reward_path = self.output_directory_path + 'reward/'
        self.rewards_np_path = self.reward_path + 'rewards.npy'
        self.esvc_path = self.output_directory_path + 'esvc/'
        os.makedirs(self.output_directory_path)
        os.makedirs(self.videodir)
        os.makedirs(self.reward_path)
        os.makedirs(self.esvc_path)
        self.rewards_file = open(self.reward_path + 'rewards.txt', "a+")
        # self.esvc_file = open(self.esvc_path + 'esvc.txt', "a+")
        # self.policy_file = open(self.irl_agent.output_directory_path + 'policy.txt', "a+")

        self.batch_size = 128
        self.batch_ids = np.zeros(self.batch_size)

        # to use in list compression
        self.cur_loop_ctr = 0

        self.mean_trajectory_length = 0
        self.emp_fc = np.zeros(self.env.num_states)
        self.calculate_emp_fc()

        # Variables used in calculations
        self.vi_loop = 120  # self.mean_trajectory_length
        self.normalized_states = np.empty_like(self.env.states)
        self.v = np.empty((len(self.env.states), self.vi_loop), dtype=float)
        self.q = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.advantage = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.fast_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.esvc = np.empty(len(self.env.states), dtype=float)
        self.esvc_mat = np.empty((len(self.env.states), self.vi_loop), dtype=float)

        self.mc_normalized_states()

        # self.q_file = open(self.output_directory_path + 'q.txt', "a+")

    ###############################################
    # [1]: Calculates the policy using an approximate version of Value Iteration
    def fast_backward_pass(self):
        # print("+ IRLAgent.backward_pass")

        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max
        q = np.zeros((len(self.env.states), len(self.env.actions)))

        goal_states = self.env.get_goal_state()

        for i in tqdm(range(self.vi_loop-1)):
            prev_v = v
            v[goal_states] = 0

            for s in range(len(self.env.states)):
                if s not in goal_states:
                    q[s, :] = np.matmul(self.env.transitions[s, :, :], v).T + self.state_rewards[s]

                q_exp = np.exp(q[s, :])
                sum_q_exp = np.sum(q_exp)
                if sum_q_exp != 0:
                    q[s, :] = q[s, :] * (q_exp/sum_q_exp)
                else:  # if sum_q_exp is 0 then q_exp is 0 so each q is -inf. Meaning, v should be -inf
                    q[s, :] = q[s, :] / self.env.num_actions + 0.01  # insane trick to get rid of -inf and have e-308
                    # because when minimum number a = e-308 --> (a/3)*3 becomes -inf
            v = np.sum(q, axis=1)

            # if i % 20 == 19:
            #     print('\rBackward Pass: {}'.format((i + 1)), end='')

            # print(np.max(np.abs(prev_v - v)))

        print('')
        # self.save_q(q, ind)
        v[goal_states] = 0
        # current MaxEnt policy:
        self.advantage = q - np.reshape(v, (self.env.num_states, 1))
        temp_policy = np.exp(self.advantage)

        self.fast_policy = np.array([temp_policy[i]/np.sum(temp_policy[i]) for i in range(len(temp_policy))])
        self.fast_policy[goal_states] = 1
        # self.plot_policy()
        # print("\n- IRLAgent.backward_pass")

    ###############################################
    # [1]: Simulates the propagation of the policy
    def fast_forward_pass(self):  # esvc: expected state visitation count
        # print("+ IRLAgent.forward_pass")

        start_states = self.env.get_start_state()
        goal_states = self.env.get_goal_state()

        self.esvc_mat[:] = 0
        self.esvc_mat[start_states, 0] = 1/len(np.atleast_1d(start_states))

        for loop_ctr in tqdm(range(self.vi_loop-1)):
            self.cur_loop_ctr = loop_ctr
            self.esvc_mat[goal_states, loop_ctr] = 0
            self.esvc_mat[:, loop_ctr + 1] = self.fast_calc_esvc_unnorm()

            # if loop_ctr % 10 == 9:
            #     print('\rForward Pass: {}'.format((loop_ctr + 1)), end='')

        print('')
        self.esvc = np.sum(self.esvc_mat, axis=1)/self.vi_loop  # averaging over <self.vi_loop> many examples
        # print("\n- IRLAgent.forward_pass")

    ###############################################

    def fast_calc_esvc_unnorm(self):
        t0 = time.time()
        esvc_arr = [self.esvcind(i) for i in range(len(self.env.states))]
        t1 = time.time()
        print(f'Took {int(round((t1-t0) * 1000))} milliseconds')
        return esvc_arr

    def esvcind(self, ind):
        esvc = np.matmul((self.env.transitions[:, :, ind] * self.fast_policy).T, self.esvc_mat[:, self.cur_loop_ctr])
        return np.sum(esvc)

    ###############################################
    ###############################################
    ###############################################
    ###############################################

    def new_backward_pass(self):
        print("IRLAgent.backward_pass")
        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max
        q = np.zeros((len(self.env.states), len(self.env.actions)))

        goal_states = self.env.get_goal_state()

        for i in tqdm(range(self.vi_loop - 1)):
            # prev_v = v
            v[goal_states] = 0

            for s in range(self.env.num_states):
                if s not in goal_states:
                    for a in range(self.env.num_actions):
                        sum_q = 0
                        for destination in self.env.forward_transitions[s, a]:
                            sum_q = self.env.transitions[s, a, destination] * v[destination]
                        q[s, a] = sum_q
                    q[s, :] += self.state_rewards[s]

                q_exp = np.exp(q[s, :])
                sum_q_exp = np.sum(q_exp)
                if sum_q_exp != 0:
                    q[s, :] = q[s, :] * (q_exp / sum_q_exp)
                else:  # if sum_q_exp is 0 then q_exp is 0 so each q is -inf. Meaning, v should be -inf
                    q[s, :] = q[s, :] / self.env.num_actions + 0.01  # insane trick to get rid of -inf and have e-308
                    # because when minimum number a = e-308 --> (a/3)*3 becomes -inf
            v = np.sum(q, axis=1)
        print('')
        # self.save_q(q, ind)
        v[goal_states] = 0
        # current MaxEnt policy:
        self.advantage = q - np.reshape(v, (self.env.num_states, 1))
        temp_policy = np.exp(self.advantage)

        self.fast_policy = np.array([temp_policy[i] / np.sum(temp_policy[i]) for i in range(len(temp_policy))])
        self.fast_policy[goal_states] = 1

    def new_forward_pass(self):  # esvc: expected state visitation count
        print("IRLAgent.forward_pass")
        start_states = self.env.get_start_state()
        goal_states = self.env.get_goal_state()

        self.esvc_mat[:] = 0
        self.esvc_mat[start_states, 0] = 1 / len(np.atleast_1d(start_states))

        for loop_ctr in tqdm(range(self.vi_loop - 1)):
            self.cur_loop_ctr = loop_ctr
            self.esvc_mat[goal_states, loop_ctr] = 0
            for s in range(self.env.num_states):
                start_action = self.env.backward_transitions[s]
                sum_esvc = 0
                for start, action in start_action:  # for each (start, action) pair that leads to state s
                    sum_esvc += self.env.transitions[start, action, s] * self.fast_policy[start, action] * \
                                self.esvc_mat[start, loop_ctr]
                self.esvc_mat[s, loop_ctr+1] = sum_esvc

        self.esvc = np.sum(self.esvc_mat, axis=1)

    ###############################################
    ###############################################
    ###############################################
    ###############################################

    def calculate_emp_fc_(self):
        cumulative_emp_fc = np.zeros_like(self.emp_fc)
        trajectories = np.load(self.path + 'trajectories_of_ids.npy', encoding='bytes', allow_pickle=True)
        for trajectory in trajectories:
            current_trajectory_emp_fc = np.zeros_like(self.emp_fc)
            for state_action in trajectory:  # state_action: [state, action]
                current_trajectory_emp_fc[state_action[0]] += 1
            current_trajectory_emp_fc /= len(trajectory)  # normalization over one trajectory
            cumulative_emp_fc += current_trajectory_emp_fc

        cumulative_emp_fc /= len(trajectories)  # normalization over all trajectories
        self.emp_fc = cumulative_emp_fc
        # self.plot_emp_fc('empfc')
        self.plot_in_state_space(self.emp_fc, path=self.output_directory_path+'empfc',
                                 title='Empirical Feature Counts')

    def calculate_emp_fc(self):
        trajectories = np.load(self.env.env_path + 'trajectories_of_ids.npy', encoding='bytes', allow_pickle=True)
        found = False
        len_traj = 0
        trajectory_lengths = []

        while not found:
            for trajectory in trajectories:
                if not found:
                    if trajectory[0][0] != self.env.start_state_id:
                        continue
                    else:
                        found = True

                        for step, state_action in enumerate(trajectory):  # state_action: [state, action]
                            self.emp_fc[state_action[0]] += 1
                            len_traj = step + 1
                        trajectory_lengths.append(len_traj)
                        break

            if not found:
                print(f"No trajectory with start state: {self.env.start_state_id}")
                # reassigning start state to match a trajectory
                self.env.start_state_id = None
                self.env.get_start_state()

        self.emp_fc /= len_traj  # normalization over all trajectories

        self.mean_trajectory_length = int(np.ceil(np.average(trajectory_lengths)))

        # self.plot_emp_fc('empfc')
        self.plot_in_state_space(self.emp_fc, path=self.output_directory_path+'empfc', title='Empirical Feature Counts')

    def policy(self, sid, aid):
        return np.exp(self.q[sid][aid] - self.v[sid, -1])   # last column in the v matrix

    def reward(self, state):
        return self.rew_nn.forward(np.asarray([state.x, state.v]))

    def reward_batch(self):
        # already shuffled random batch
        self.batch_ids = np.random.choice(len(self.state_rewards), self.batch_size, replace=False)
        rew_batch = self.rew_nn.forward_batch(self.normalized_states[self.batch_ids].tolist())
        self.state_rewards[self.batch_ids] = rew_batch[:, 0]

    def backpropagation_batch(self, dist, lr):
        self.rew_nn.backprop_diff(dist[self.batch_ids].tolist(), self.normalized_states[self.batch_ids].tolist(),
                                  self.state_rewards[self.batch_ids], lr)

    def initialize_rewards(self):
        self.state_rewards = np.random.rand(len(self.state_rewards)) * 2 - 1

    def mc_normalized_states(self):
        normalized_states = self.env.states.copy()

        min0 = np.min(normalized_states[:, 0])
        min1 = np.min(normalized_states[:, 1])
        max0 = np.max(normalized_states[:, 0])
        max1 = np.max(normalized_states[:, 1])

        normalized_states -= [min0, min1]
        normalized_states /= [max0-min0, max1-min1]

        self.normalized_states = normalized_states * 2 - 1  # scaling the states between [0, 1]

    def plot_reward(self, nof_iter):
        self.plot_in_state_space(self.state_rewards, ind=nof_iter, path=self.reward_path, title='State Rewards')

    def plot_in_state_space(self, inp, ind=-1, path="", xlabel='x', ylabel='v', title=''):
        if path == "":
            path = self.output_directory_path
        else:
            if ind != -1:
                path = path+str(ind)
        data = np.reshape(inp, self.env.shape)
        plt.figure(figsize=(20, 14))
        hm = sb.heatmap(data.T, linewidths=0.025, linecolor='silver')
        hm.set_title(title)
        hm.set_xlabel(xlabel)
        hm.set_ylabel(ylabel)
        fig = hm.get_figure()
        fig.savefig(path + '.png')
        fig.clf()

    # def plot_in_t_state_space(self, inp, ind=-1, path="", xlabel='x', ylabel='v', zlabel='t', title=''):
    #     if path == "":
    #         path = self.output_directory_path
    #     else:
    #         if ind != -1:
    #             path = path+str(ind)
    #     data = np.reshape(inp, self.env.t_shape)
    #
    #     z = []
    #     x, y = [], []
    #     for step in range(self.env.t_div):
    #         if np.any(data[step] > 0):
    #             xi, yi = np.where(data[step] != 0)
    #             x.append(xi[0])
    #             y.append(yi[0])
    #             z.append(step)
    #         else:
    #             break
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     ax.scatter(x, y, z, c='r', marker='o')
    #
    #     ax.set_title(title)
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_zlabel(zlabel)
    #     plt.savefig(path + '.png')
    #     fig.clf()

    def save_reward(self, nof_iter):
        self.rewards_file.write(str(nof_iter) + "\n")
        self.rewards_file.write("[")

        for i, r in enumerate(self.state_rewards):
            self.rewards_file.write(str(r))
            if i != len(self.state_rewards)-1:
                self.rewards_file.write(", ")

        self.rewards_file.write("] \n")
        self.rewards_file.flush()

        # saving the last rewards, updating every iteration
        np.save(self.rewards_np_path, self.state_rewards)

    # def save_q(self, q, ind):
    #     self.q_file.write(str(ind) + "\n")
    #     self.q_file.write("[")
    #
    #     for i in range(len(self.env.states)):
    #         self.q_file.write("[")
    #         for j in range(len(self.env.actions)):
    #             self.q_file.write(str(q[i, j]))
    #             if j != len(self.env.actions) - 1:
    #                 self.q_file.write(", ")
    #         self.q_file.write("]")
    #         if i != len(self.env.states) - 1:
    #             self.q_file.write(", ")
    #
    #     self.q_file.write("] \n\n")
    #     self.q_file.flush()

    def run_policy(self, str_id):
        outdir = self.videodir + str_id + "/"
        os.makedirs(outdir)
        env = gym.make('MountainCarContinuous-v0')
        env = gym.wrappers.Monitor(env, outdir, video_callable=lambda episode_id: True)
        done = False
        step_ctr = 0

        s = env.reset()
        current_s = self.env.find_closest_states(s)[0]
        while not done and step_ctr < 3500:
            env.render()
            action_id = np.random.choice(range(len(self.env.actions)), 1, self.fast_policy[current_s, :].tolist())[0]
            # action_id = np.argmax(self.fast_policy[current_s, :])
            next_s, _, done, _ = env.step(np.array([self.env.actions[action_id]]))
            current_s = self.env.find_closest_states(next_s)[0]
            step_ctr += 1
            # time.sleep(0.01)
            # print("State: ", current_s, " - Action: ", self.env.actions[action_id].force)

        env.close()
