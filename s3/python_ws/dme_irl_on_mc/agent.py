"""
  @author: yigit.yildirim@boun.edu.tr

  [1]: Kitani 2012, Activity Forecasting
  [2]: Wulfmeier 2016, Maximum Entropy Deep Inverse Reinforcement Learning
"""
import numpy as np
from env import IRLMDP
from neural_network import MyNN, sigm, linear

import sys
import seaborn as sb
import os
import time


class IRLAgent:
    def __init__(self):
        self.env = IRLMDP()
        # initializes nn with random weights
        self.rew_nn = MyNN(nn_arch=(2, 400, 300, 1), acts=[sigm, sigm, linear])
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
        self.current_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.fast_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.esvc = np.empty(len(self.env.states), dtype=float)
        self.esvc_mat = np.empty((len(self.env.states), self.vi_loop), dtype=float)

        # to use in list compression
        self.cur_loop_ctr = 0

        self.emp_fc = np.zeros(len(self.env.states))
        self.calculate_emp_fc()

    ###############################################
    # [1]
    def backward_pass(self):
        # print("+ IRLAgent.backward_pass")

        self.v[:] = -sys.float_info.max

        for i in range(self.vi_loop-1):
            self.v[self.env.goal_id, i] = 0

            for s in range(len(self.env.states)):
                if s == self.env.goal_id:
                    continue
                state_reward = self.state_rewards[s]
                for a in range(len(self.env.actions)):
                    self.q[s][a] = state_reward + np.matmul(self.env.transition[s][a], self.v[:, i])

            # v  = softmax_a Q
            q = np.exp(self.q)
            max_q = np.max(q, axis=1)   # max of each row: max q value over actions for each state
            nonzero_ids = np.where(max_q != 0)
            self.v[nonzero_ids, i+1] = max_q[nonzero_ids] / np.sum(q[nonzero_ids], axis=1)

            print('\rBackward Pass: {}'.format((i+1)), end='')
        print('')

        # self.plot_policy()
        # print("\n- IRLAgent.backward_pass")

    ###############################################
    # [1]: Calculates fast_policy using an approximate version of Value Iteration
    def fast_backward_pass(self):
        # print("+ IRLAgent.backward_pass")

        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max
        q = np.zeros((len(self.env.states), len(self.env.actions)))

        for i in range(self.vi_loop-1):
            v[self.env.goal_id] = 0
            for s in range(len(self.env.states)):
                q[s, :] = np.matmul(self.env.transition[s, :, :], v).T + self.state_rewards[s]

            # v = softmax_a q
            # one problem:
            # when np.sum(np.exp(q), axis=1) = 0, division by 0. In this case v = 0
            expq = np.exp(q)
            sumexpq = np.sum(expq, axis=1)
            nonzero_ids = np.where(sumexpq != 0)
            zero_ids = np.where(sumexpq == 0)
            v[nonzero_ids, 0] = np.exp(np.max(q[nonzero_ids], axis=1))/sumexpq[nonzero_ids]
            v[zero_ids, 0] = -sys.float_info.max

            print('\rBackward Pass: {}'.format((i+1)), end='')

        v[self.env.goal_id] = 0
        # current MaxEnt policy:
        self.advantage = q - np.reshape(v, (len(self.env.states), 1))
        self.fast_policy = np.exp(self.advantage)

        # self.plot_policy()
        print("\n- IRLAgent.backward_pass")

    ##############################################################################################
    # [1]: Calculates fast_policy using an approximate version of Value Iteration algorithm
    def ffast_backward_pass(self):
        # print("+ IRLAgent.backward_pass")

        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max

        for i in range(self.vi_loop-1):
            v[self.env.goal_id] = 0
            q = [(np.matmul(self.env.transition[s, :, :], v).T + self.state_rewards[s])[0]
                 for s in range(len(self.env.states))]

            # q = np.reshape(tq, (len(self.env.states), len(self.env.actions)))
            # v = softmax_a q
            # one problem:
            # when np.sum(np.exp(q), axis=1) = 0, division by 0. In this case v = 0
            expq = np.exp(q)
            sumexpq = np.sum(expq, axis=1)
            nonzero_ids = np.where(sumexpq != 0)
            zero_ids = np.where(sumexpq == 0)
            v[nonzero_ids, 0] = np.exp(np.max(np.array(q)[nonzero_ids], axis=1))/sumexpq[nonzero_ids]
            v[zero_ids, 0] = -sys.float_info.max

            print('\rBackward Pass: {}'.format((i+1)), end='')

        v[self.env.goal_id] = 0
        # current MaxEnt policy:
        self.advantage = q - np.reshape(v, (len(self.env.states), 1))
        self.fast_policy = np.exp(self.advantage)

        # self.plot_policy()
        print("\n- IRLAgent.backward_pass")

    ###############################################
    # [1]
    # def backward_pass_wo_transition_matrix(self):
    #     self.v[:] = -np.inf
    #     for i in range(self.vi_loop):
    #         self.v[self.env.goal_id] = 0

    ###############################################

    ###############################################
    # [1]
    # Simulates the propagation of the policy
    def forward_pass(self):  # esvc: expected state visitation count
        # print("+ IRLAgent.forward_pass")

        # #######################################################################
        # create the directory to be used for plotting
        # since forward will be called on multiple times, I use system time here
        path = self.env.path + 'figures/forward_pass/' + str(int(time.time()))
        os.makedirs(path)
        # #######################################################################

        self.esvc_mat[:] = 0
        self.esvc_mat[self.env.start_id, :] = 1
        # for loop_ctr in range(10):
        for loop_ctr in range(self.vi_loop-1):
            self.esvc_mat[self.env.goal_id][loop_ctr] = 0
            esvc_unnorm = np.zeros(len(self.env.states))
            for j in range(len(self.env.states)):
                esvc_unnorm[j] = self.calc_esvc_unnorm(j, loop_ctr)

            # normalization to calculate the frequencies
            self.esvc_mat[:, loop_ctr + 1] = esvc_unnorm/sum(esvc_unnorm)
            print('\rForward Pass: {}'.format((loop_ctr+1)), end='')
            self.plot_esvc_mat(path, loop_ctr)
        self.esvc = np.sum(self.esvc_mat, axis=1)
        self.plot_esvc(path, 'esvc', self.esvc)
        print('')
        print("\n- IRLAgent.forward_pass")

    ###############################################

    ###############################################
    # [1]
    def fast_forward_pass(self):  # esvc: expected state visitation count
        # print("+ IRLAgent.forward_pass")

        # #######################################################################
        # create the directory to be used for plotting
        # since forward will be called on multiple times, I use system time here
        # path = self.env.path + 'figures/forward_pass/' + str(int(time.time()))
        # os.makedirs(path)
        # #######################################################################

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
        self.esvc = np.sum(self.esvc_mat, axis=1)/self.vi_loop  # averaging over <self.vi_loop> many examples
        # self.plot_esvc(path, 'esvc', self.esvc)
        # print('')
        print("\n- IRLAgent.forward_pass")

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
        # esvc = map(self.esvcind, range(len(self.env.states)))
        esvc = [self.esvcind(i) for i in range(len(self.env.states))]

        return np.sum(esvc, axis=0)

    def esvcind(self, ind):
        return np.matmul(self.env.transition[ind][:][:].T, self.fast_policy[ind][:].T) \
               * self.esvc_mat[ind][self.cur_loop_ctr]

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

    # def exp_fc(self):   # expected feature counts
    #     # return np.matmul(self.esvc.T, self.env.state_list)
    #     return self.esvc

    def policy(self, sid, aid):
        return np.exp(self.q[sid][aid] - self.v[sid, -1])   # last column in the v matrix

    def set_current_policy(self):
        self.current_policy = np.exp(self.q - np.reshape(self.v[:, -1], (len(self.env.states), 1)))

    def reward(self, state):
        return self.rew_nn.forward(np.asarray([state.x, state.v]))

    def reward_batch(self):
        return self.rew_nn.forward_batch(np.asarray(self.mc_normalized_states()))

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
