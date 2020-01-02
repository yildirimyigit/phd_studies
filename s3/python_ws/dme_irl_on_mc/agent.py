"""
  @author: yigit.yildirim@boun.edu.tr

  [1]: Kitani 2012, Activity Forecasting
  [2]: Wulfmeier 2016, Maximum Entropy Deep Inverse Reinforcement Learning
"""
import numpy as np
from env import IRLMDP
from neural_network import MyNN, sigm, linear, relu, tanh

import sys
import seaborn as sb
import os
import time


class IRLAgent:
    def __init__(self):
        self.env = IRLMDP()
        # initializes nn with random weights
        self.rew_nn = MyNN(nn_arch=(2, 32, 64, 128, 128, 64, 32, 1), acts=[sigm, sigm, relu, tanh, sigm, sigm, linear])
        self.state_rewards = np.empty(len(self.env.states), dtype=float)
        self.initialize_rewards()

        # To output the results, the following are used
        self.output_directory_suffix = str(int(time.time()))
        self.output_directory_path = self.env.path + 'output/' + self.output_directory_suffix + "/"
        # Creating the output directory for the individual run
        os.makedirs(self.output_directory_path)

        self.vi_loop = 150
        self.normalized_states = np.empty(len(self.env.states))
        self.v = np.empty((len(self.env.states), self.vi_loop), dtype=float)
        self.q = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.advantage = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.fast_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.esvc = np.empty(len(self.env.states), dtype=float)
        self.esvc_mat = np.empty((len(self.env.states), self.vi_loop), dtype=float)

        self.batch_size = 128
        self.batch_ids = np.zeros(self.batch_size)

        # to use in list compression
        self.cur_loop_ctr = 0

        self.emp_fc = np.zeros(len(self.env.states))
        self.calculate_emp_fc()

        self.mc_normalized_states()

        self.q_file = open(self.output_directory_path + 'q.txt', "a+")

    ###############################################
    # [1]: Calculates the policy using an approximate version of Value Iteration
    def fast_backward_pass(self, ind):
        # print("+ IRLAgent.backward_pass")

        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max
        q = np.zeros((len(self.env.states), len(self.env.actions)))

        for i in range(self.vi_loop-1):
            v[self.env.goal_id] = 0
            for s in range(len(self.env.states)):
                q[s, :] = np.matmul(self.env.transition[s, :, :], v).T + self.state_rewards[s]

            # v = softmax_a q
            # one problem: when np.sum(np.exp(q), axis=1) = 0, division by 0. In this case v = 0
            # expq = np.exp(q)
            # sumexpq = np.sum(expq, axis=1)
            # nonzero_ids = np.where(sumexpq != 0)
            # zero_ids = np.where(sumexpq == 0)
            # v[nonzero_ids, 0] = np.exp(np.max(q[nonzero_ids], axis=1))/sumexpq[nonzero_ids]
            # v[zero_ids, 0] = -sys.float_info.max
            v = np.max(q, axis=1)

            if i % 20 == 19:
                print('\rBackward Pass: {}'.format((i + 1)), end='')

        print('')
        # self.save_q(q, ind)
        v[self.env.goal_id] = 0
        # current MaxEnt policy:
        self.advantage = q - np.reshape(v, (len(self.env.states), 1))
        temp_policy = np.exp(self.advantage)

        self.fast_policy = np.array([temp_policy[i]/np.sum(temp_policy[i]) for i in range(len(temp_policy))])
        self.fast_policy[self.env.goal_id] = 0
        # self.plot_policy()
        # print("\n- IRLAgent.backward_pass")

    ###############################################
    # [1]: Simulates the propagation of the policy
    def fast_forward_pass(self):  # esvc: expected state visitation count
        # print("+ IRLAgent.forward_pass")

        self.esvc_mat[:] = 0
        self.esvc_mat[self.env.start_id, :] = 1
        for loop_ctr in range(self.vi_loop-1):
            self.cur_loop_ctr = loop_ctr
            self.esvc_mat[self.env.goal_id][loop_ctr] = 0
            self.esvc_mat[:, loop_ctr + 1] = self.ffast_calc_esvc_unnorm()

            if loop_ctr % 20 == 19:
                print('\rForward Pass: {}'.format((loop_ctr + 1)), end='')

        print('')
        self.esvc = np.sum(self.esvc_mat, axis=1)/self.vi_loop  # averaging over <self.vi_loop> many examples
        # self.plot_esvc(path, 'esvc', self.esvc)
        # print("\n- IRLAgent.forward_pass")

    ###############################################

    # calculation of the unnormalized esvc for state 'index'
    def calc_esvc_unnorm(self, index, loop_ctr):
        sum_esvc = 0
        for i in range(len(self.env.states)):
            for j in range(len(self.env.actions)):
                sum_esvc += self.env.transition[i][j][index] * self.policy(i, j) * self.esvc_mat[i, loop_ctr]
        return sum_esvc

    ###############################################

    # calculation of the unnormalized esvc for state 'index'
    def fast_calc_esvc_unnorm(self, loop_ctr):
        esvc = np.zeros((len(self.env.states), len(self.env.states)))

        for i in range(len(self.env.states)):
            esvc[:, i] = np.matmul(self.env.transition[i][:][:].T, self.fast_policy[i][:].T) \
                         * self.esvc_mat[i][loop_ctr]

        return np.sum(esvc, axis=1)

    ###############################################

    def ffast_calc_esvc_unnorm(self):
        esvc_arr = [self.esvcind(i) for i in range(len(self.env.states))]
        return esvc_arr

    def esvcind(self, ind):
        esvc = np.matmul((self.env.transition[:, :, ind] * self.fast_policy).T, self.esvc_mat[:, self.cur_loop_ctr])
        return np.sum(esvc)

    ###############################################

    def calculate_emp_fc(self):
        cumulative_emp_fc = np.zeros_like(self.emp_fc)
        trajectories = np.load(self.env.path + 'trajectories_of_ids.npy', encoding='bytes', allow_pickle=True)
        for trajectory in trajectories:
            current_trajectory_emp_fc = np.zeros_like(self.emp_fc)
            for state_action in trajectory:  # state_action: [state, action]
                current_trajectory_emp_fc[state_action[0]] += 1
            current_trajectory_emp_fc /= len(trajectory)  # normalization over one trajectory
            cumulative_emp_fc += current_trajectory_emp_fc

        cumulative_emp_fc /= len(trajectories)  # normalization over all trajectories
        self.emp_fc = cumulative_emp_fc
        self.plot_emp_fc('empfc')

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
        self.rew_nn.backprop_diff(dist[self.batch_ids].tolist(), np.array(self.env.state_list)[self.batch_ids].tolist(),
                                  self.state_rewards[self.batch_ids], lr)

    def initialize_rewards(self):
        self.state_rewards = np.random.rand(len(self.state_rewards)) * 2 - 1

    def mc_normalized_states(self):
        normalized_states = np.asarray(self.env.state_list)

        min0 = np.min(normalized_states[:, 0])
        min1 = np.min(normalized_states[:, 1])
        max0 = np.max(normalized_states[:, 0])
        max1 = np.max(normalized_states[:, 1])

        normalized_states -= [min0, min1]
        normalized_states /= [max0-min0, max1-min1]

        self.normalized_states = normalized_states * 2 - 1

    def plot_esvc(self, path, name, data):
        dim = int(np.sqrt(len(self.env.state_list)))
        hm = sb.heatmap(np.reshape(data, (dim, dim)))
        fig = hm.get_figure()
        fig.savefig(path+'/' + name + '.png')
        fig.clf()

    def plot_emp_fc(self, name):
        dim = int(np.sqrt(len(self.env.state_list)))
        hm = sb.heatmap(np.reshape(self.emp_fc, (dim, dim)).T)
        hm.set_title('Empirical Feature Counts')
        hm.set_xlabel('x')
        hm.set_ylabel('velocity')
        fig = hm.get_figure()
        fig.savefig(self.output_directory_path + name + '.png')
        fig.clf()

    def plot_esvc_mat(self, path, i):
        dim = int(np.sqrt(len(self.env.state_list)))
        hm = sb.heatmap(np.reshape(self.esvc_mat[:, -1], (dim, dim)).T)
        hm.set_title('Expected State Visitation Counts')
        hm.set_xlabel('x')
        hm.set_ylabel('velocity')
        fig = hm.get_figure()
        fig.savefig(path + "esvc_" + str(i) + '.png')
        fig.clf()

    def save_q(self, q, ind):
        self.q_file.write(str(ind) + "\n")
        self.q_file.write("[")

        for i in range(len(self.env.states)):
            self.q_file.write("[")
            for j in range(len(self.env.actions)):
                self.q_file.write(str(q[i, j]))
                if j != len(self.env.actions) - 1:
                    self.q_file.write(", ")
            self.q_file.write("]")
            if i != len(self.env.states) - 1:
                self.q_file.write(", ")

        self.q_file.write("] \n\n")
        self.q_file.flush()
