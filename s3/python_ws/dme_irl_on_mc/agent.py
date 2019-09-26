"""
  @author: yigit.yildirim@boun.edu.tr

  [1]: Kitani 2012, Activity Forecasting
"""
import numpy as np
from env import IRLMDP
from neural_network import MyNN, sigm, linear, tanh, gaussian, relu, elu

import sys
import seaborn as sb
import os
import time


# class RLAgent:
#     def __init__(self):
#         self.max_episode_steps = 1000
#         self.learning_rate = 0.9
#         self.epsilon = 0.25
#         self.max_episode = 100000
#         self.episode = 0
#         self.episode_reward = 0
#         self.episode_steps = 0
#
#         self.env = MDP()
#         self.state_id = self.env.get_start_state()
#         self.episode_rewards = np.zeros((self.max_episode, 1))
#
#         nof_actions = self.env.actions.shape[0]
#         nof_states = self.env.states.shape[0]
#         self.q_table = np.zeros((nof_states, nof_actions))
#
#     def act(self, action):
#         self.state_id, reward, goal_reached = self.env.step(self.env.states[self.state_id], action)
#         self.episode_steps += 1
#         self.episode_reward += reward
#         if goal_reached or self.episode_steps >= self.max_episode_steps:
#             self.restart_episode()
#             self.episode += 1
#             return reward, True
#         return reward, False
#
#     def restart_episode(self):
#         self.episode_rewards[self.episode] = self.episode_reward
#         print('Episode: {0} - Reward: {1}'.format(self.episode, self.episode_reward))
#         self.episode_reward = 0
#         self.episode_steps = 0
#         self.state_id = self.env.get_start_state()
#
#     def initialize_q(self):
#         self.q_table = np.random.rand(self.q_table.shape) / 100.0
#
#     def q_learn(self):
#         self.initialize_q()
#
#         while self.episode < self.max_episode:
#             episode_end = False
#             while not episode_end:
#                 chosen_action_id = self.choose_action()
#                 p_sid = self.state_id
#                 c_reward, episode_end = self.act(chosen_action_id)
#                 self.q_table[p_sid][chosen_action_id] += self.learning_rate * (c_reward + self.env.gamma * np.max(
#                     self.q_table[self.state_id][:]) - self.q_table[p_sid][chosen_action_id])
#
#     def choose_action(self):
#         if np.random.rand() < self.get_epsilon():  # epsilon-greedy
#             chosen_action = np.random.choice(range(self.env.actions.shape[0]))
#         else:
#             chosen_action = np.argmax(self.q_table[self.state_id][:])
#         return chosen_action
#
#     def get_epsilon(self):  # decaying epsilon
#         return np.max([0.05, self.epsilon*(1-(self.episode/self.max_episode))])


class IRLAgent:
    def __init__(self):
        self.env = IRLMDP()
        # initializes nn with random weights
        self.rew_nn = MyNN(nn_arch=(2, 400, 300, 1), acts=[sigm, sigm, linear])
        self.state_rewards = np.empty(len(self.env.states))

        # self.state_id = self.env.start_id

        self.vi_loop = 2500
        self.v = np.empty((len(self.env.states), self.vi_loop), dtype=float)
        self.q = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.current_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.fast_policy = np.empty((len(self.env.states), len(self.env.actions)), dtype=float)
        self.esvc = np.empty(len(self.env.states), dtype=float)
        self.esvc_mat = np.empty((len(self.env.states), self.vi_loop), dtype=float)

        self.emp_fc = 0
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
            v[nonzero_ids, 0] = np.exp(np.max(q[nonzero_ids], axis=1))/np.sum(expq[nonzero_ids], axis=1)
            v[zero_ids, 0] = -sys.float_info.max

            print('\rBackward Pass: {}'.format((i+1)), end='')

        v[self.env.goal_id] = 0
        # current MaxEnt policy:
        self.fast_policy = np.exp(q - np.reshape(v, (len(self.env.states), 1)))

        # self.plot_policy()
        # print("\n- IRLAgent.backward_pass")

    ###############################################

    ###############################################
    # [1]
    # def backward_pass_wo_transition_matrix(self):
    #     self.v[:] = -np.inf
    #     for i in range(self.vi_loop):
    #         self.v[self.env.goal_id] = 0

    ###############################################

    ###############################################
    # [1]
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

            # normalization to calculate the frequencies. Rounding just because.
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
            self.esvc_mat[self.env.goal_id][loop_ctr] = 0
            esvc_unnorm = self.fast_calc_esvc_unnorm(loop_ctr)

            # normalization to calculate the frequencies.
            self.esvc_mat[:, loop_ctr + 1] = esvc_unnorm/sum(esvc_unnorm)
            print('\rForward Pass: {}'.format((loop_ctr+1)), end='')
            # self.plot_esvc_mat(path, loop_ctr)
        self.esvc = np.sum(self.esvc_mat, axis=1)
        # self.plot_esvc(path, 'esvc', self.esvc)
        # print('')
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

    def calculate_emp_fc(self):
        trajectories = np.load(self.env.path + 'trajectories.npy', encoding='bytes', allow_pickle=True)
        sum_traj_feats = []
        for trajectory in trajectories:
            sum_traj_feats.append(np.sum(trajectory, axis=0))

        sum_all_feats = np.sum(sum_traj_feats, axis=0)
        self.emp_fc = sum_all_feats/len(trajectories)
        # self.plot_esvc('data/figures/forward_pass', 'empfc', self.emp_fc)

    def exp_fc(self):   # expected feature counts
        return np.matmul(self.esvc.T, self.env.state_list)

    def policy(self, sid, aid):
        return np.exp(self.q[sid][aid] - self.v[sid, -1])   # last column in the v matrix

    def set_current_policy(self):
        self.current_policy = np.exp(self.q - np.reshape(self.v[:, -1], (len(self.env.states), 1)))

    def reward(self, state):
        return self.rew_nn.forward(np.asarray([state.x, state.v]))

    def reward_batch(self):
        return self.rew_nn.forward_batch(np.asarray(self.env.state_list))

    def plot_esvc(self, path, name, data):
        dim = int(np.sqrt(len(self.env.state_list)))
        hm = sb.heatmap(np.reshape(data, (dim, dim)))
        fig = hm.get_figure()
        fig.savefig(path+'/' + name + '.png')
        fig.clf()

    def plot_esvc_mat(self, path, i):
        hm = sb.heatmap(self.esvc_mat)
        fig = hm.get_figure()
        fig.savefig(path+'/Figure' + str(i) + '.png')
        fig.clf()
