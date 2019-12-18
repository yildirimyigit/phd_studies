"""
  @author: yigit.yildirim@boun.edu.tr

  [1]: Kitani 2012, Activity Forecasting
  [2]: Wulfmeier 2016, Maximum Entropy Deep Inverse Reinforcement Learning
"""
import numpy as np
import torch
from env import IRLMDP
# from neural_network import MyNN, relu, tanh
from nn import NN

import sys
import seaborn as sb
import os
import time


class IRLAgent:
    def __init__(self):
        self.env = IRLMDP()
        # initializes nn with random weights
        self.nn = NN(nn_arch=[2, 16, 16, 1]).double()
        self.state_rewards = np.empty(len(self.env.states), dtype=float)

        # self.state_id = self.env.start_id

        # To output the results, the following are used
        self.output_directory_suffix = str(int(time.time()))
        self.output_directory_path = self.env.path + 'output/' + self.output_directory_suffix + "/"
        # Creating the output directory for the individual run
        os.makedirs(self.output_directory_path)

        self.vi_loop = 1000
        self.v = np.empty((len(self.env.states), self.vi_loop), dtype=float)
        self.q = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.advantage = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.fast_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.esvc = torch.empty(len(self.env.states), dtype=float)
        self.esvc_mat = torch.empty((len(self.env.states), self.vi_loop), dtype=float)

        # to use in list compression
        self.cur_loop_ctr = 0

        self.emp_fc = np.zeros(len(self.env.states))
        self.calculate_emp_fc()

        self.normalized_states = self.mc_normalized_states()

    ###############################################
    # [1]: Calculates fast_policy using an approximate version of Value Iteration
    def fast_backward_pass(self, ordered_rewards):
        # print("+ IRLAgent.backward_pass")

        v = torch.ones((len(self.env.states), 1)).double() * -1e30
        q = torch.zeros((len(self.env.states), len(self.env.actions))).double()

        for i in range(self.vi_loop-1):
            v[self.env.goal_id] = 0
            for s in range(len(self.env.states)):
                q[s, :] = torch.matmul(self.env.transition[s, :, :], v).T + ordered_rewards[s]

            # v = softmax_a(q)
            # one problem:
            # when np.sum(np.exp(q), axis=1) = 0, division by 0. In this case v = 0
            expq = torch.exp(q)
            sumexpq = torch.sum(expq, dim=1)
            nonzero_ids = torch.where(sumexpq != 0)[0]
            zero_ids = torch.where(sumexpq == 0)[0]
            v[nonzero_ids, 0] = torch.exp(torch.max(q[nonzero_ids], dim=1)[0])/sumexpq[nonzero_ids]
            v[zero_ids, 0] = -1e30

            print('\rBackward Pass: {}'.format((i+1)), end='')
        print('')
        v[self.env.goal_id] = 0
        # current MaxEnt policy:
        self.advantage = q - v.view(-1, 1)
        self.fast_policy = torch.exp(self.advantage)

    ###############################################
    # [1]
    # Simulates the propagation of the policy
    def fast_forward_pass(self):  # esvc: expected state visitation count
        # print("+ IRLAgent.forward_pass")

        self.esvc_mat[:] = 0
        self.esvc_mat[self.env.start_id, :] = 1
        # for i in range(10):
        for loop_ctr in range(self.vi_loop-1):  # type: int
            self.cur_loop_ctr = loop_ctr
            self.esvc_mat[self.env.goal_id][loop_ctr] = 0
            # esvc_unnorm = self.fast_calc_esvc_unnorm(loop_ctr)
            esvc_unnorm = self.ffast_calc_esvc_unnorm()

            # normalization to calculate the frequencies.
            self.esvc_mat[:, loop_ctr + 1] = esvc_unnorm/sum(esvc_unnorm)
            print('\rForward Pass: {}'.format((loop_ctr+1)), end='')
            # self.plot_esvc_mat(path, loop_ctr)
        print('')
        self.esvc = torch.sum(self.esvc_mat, dim=1)/self.vi_loop  # averaging over <self.vi_loop> many examples
        # self.plot_esvc(path, 'esvc', self.esvc)
        # print('')
        # print("\n- IRLAgent.forward_pass")

    ###############################################

    def ffast_calc_esvc_unnorm(self):
        # esvc = map(self.esvcind, range(len(self.env.states)))
        esvc = [self.esvcind(i) for i in range(len(self.env.states))]

        return np.sum(esvc, axis=0)

    def esvcind(self, ind):
        propagation_prob = torch.matmul(self.env.transition[ind][:][:].T, self.fast_policy[ind][:].T)
        esvc = propagation_prob * self.esvc_mat[ind][self.cur_loop_ctr]
        return esvc

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

    def get_rewards(self):
        states, ids = self.get_states()  # states are shuffled, ids contains the order
        self.nn.zero_grad()
        out = self.nn(states.view(-1, 2))
        return out, ids

    def policy(self, sid, aid):
        return np.exp(self.q[sid][aid] - self.v[sid, -1])   # last column in the v matrix

    def get_states(self):
        indices = np.array(range(len(self.normalized_states)))
        np.random.shuffle(indices)
        return torch.from_numpy(np.array(self.normalized_states)[indices]), indices

    def order_rewards(self, ids):
        ordered_rewards = torch.zeros_like(self.state_rewards)
        for i in range(len(ordered_rewards)):
            ordered_rewards[i] = self.state_rewards[np.where(ids == i)]

        return ordered_rewards

    def mc_normalized_states(self):
        normalized_states = np.asarray(self.env.state_list)

        min0 = np.min(normalized_states[:, 0])
        min1 = np.min(normalized_states[:, 1])
        max0 = np.max(normalized_states[:, 0])
        max1 = np.max(normalized_states[:, 1])

        normalized_states -= [min0, min1]
        normalized_states /= [max0-min0, max1-min1]

        a = normalized_states.tolist()

        return a

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
        hm = sb.heatmap(self.esvc_mat)
        fig = hm.get_figure()
        fig.savefig(path+'/Figure' + str(i) + '.png')
        fig.clf()
