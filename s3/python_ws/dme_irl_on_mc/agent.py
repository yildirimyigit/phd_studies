"""
  @author: yigit.yildirim@boun.edu.tr

  [1]: Kitani 2012, Activity Forecasting
  [2]: Wulfmeier 2016, Maximum Entropy Deep Inverse Reinforcement Learning
"""
import numpy as np
from mccont_mdp import MCContMDP
from neural_network import MyNN, sigm, linear, gaussian

import sys
import seaborn as sb
import os
import time

import gym
from utils import *


class IRLAgent:
    def __init__(self):
        self.env = MCContMDP()
        # initializes nn with random weights
        self.rew_nn = MyNN(nn_arch=(2, 8, 16, 32, 64, 16, 32, 16, 64, 32, 16, 8, 1),
                           acts=[sigm, sigm, sigm, sigm, sigm, sigm, sigm, gaussian, sigm, sigm, sigm, linear])
        self.state_rewards = np.empty(len(self.env.states), dtype=float)
        self.initialize_rewards()

        # To output the results, the following are used
        self.path = 'data/mccont/'
        self.output_directory_suffix = str(int(time.time()))
        # Creating the output directory for the individual run
        self.output_directory_path = self.path + 'output/' + self.output_directory_suffix + "/"
        self.videodir = self.output_directory_path + "video/"
        self.reward_path = self.output_directory_path + 'reward/'
        self.esvc_path = self.output_directory_path + 'esvc/'
        os.makedirs(self.output_directory_path)
        os.makedirs(self.videodir)
        os.makedirs(self.reward_path)
        os.makedirs(self.esvc_path)
        self.rewards_file = open(self.reward_path + 'rewards.txt', "a+")
        # self.esvc_file = open(self.esvc_path + 'esvc.txt', "a+")
        # self.policy_file = open(self.irl_agent.output_directory_path + 'policy.txt', "a+")

        # Variables used in calculations
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

        # self.q_file = open(self.output_directory_path + 'q.txt', "a+")

    ###############################################
    # [1]: Calculates the policy using an approximate version of Value Iteration
    def fast_backward_pass(self):
        # print("+ IRLAgent.backward_pass")

        v = np.ones((len(self.env.states), 1)) * -sys.float_info.max
        q = np.zeros((len(self.env.states), len(self.env.actions)))

        goal_states = self.env.get_goal_state()

        for i in range(self.vi_loop-1):
            v[goal_states] = 0
            for s in range(len(self.env.states)):
                q[s, :] = np.matmul(self.env.transitions[s, :, :], v).T + self.state_rewards[s]

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
        v[goal_states] = 0
        # current MaxEnt policy:
        self.advantage = q - np.reshape(v, (len(self.env.states), 1))
        temp_policy = np.exp(self.advantage)

        self.fast_policy = np.array([temp_policy[i]/np.sum(temp_policy[i]) for i in range(len(temp_policy))])
        self.fast_policy[goal_states] = 0
        # self.plot_policy()
        # print("\n- IRLAgent.backward_pass")

    ###############################################
    # [1]: Simulates the propagation of the policy
    def fast_forward_pass(self):  # esvc: expected state visitation count
        # print("+ IRLAgent.forward_pass")

        start_states = self.env.get_start_state()
        goal_states = self.env.get_goal_state()

        self.esvc_mat[:] = 0
        self.esvc_mat[start_states, :] = 1
        for loop_ctr in range(self.vi_loop-1):
            self.cur_loop_ctr = loop_ctr
            self.esvc_mat[goal_states][loop_ctr] = 0
            self.esvc_mat[:, loop_ctr + 1] = self.fast_calc_esvc_unnorm()

            if loop_ctr % 20 == 19:
                print('\rForward Pass: {}'.format((loop_ctr + 1)), end='')

        print('')
        self.esvc = np.sum(self.esvc_mat, axis=1)/self.vi_loop  # averaging over <self.vi_loop> many examples
        # self.plot_esvc(path, 'esvc', self.esvc)
        # print("\n- IRLAgent.forward_pass")

    ###############################################

    def fast_calc_esvc_unnorm(self):
        esvc_arr = [self.esvcind(i) for i in range(len(self.env.states))]
        return esvc_arr

    def esvcind(self, ind):
        esvc = np.matmul((self.env.transitions[:, :, ind] * self.fast_policy).T, self.esvc_mat[:, self.cur_loop_ctr])
        return np.sum(esvc)

    ###############################################

    def calculate_emp_fc(self):
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
        self.plot_in_state_space(self.emp_fc, path=self.output_directory_path+'empfc.png')

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
        self.rew_nn.backprop_diff(dist[self.batch_ids].tolist(), self.env.states[self.batch_ids].tolist(),
                                  self.state_rewards[self.batch_ids], lr)

    def initialize_rewards(self):
        self.state_rewards = np.random.rand(len(self.state_rewards)) * 2 - 1

    def mc_normalized_states(self):
        normalized_states = self.env.states

        min0 = np.min(normalized_states[:, 0])
        min1 = np.min(normalized_states[:, 1])
        max0 = np.max(normalized_states[:, 0])
        max1 = np.max(normalized_states[:, 1])

        normalized_states -= [min0, min1]
        normalized_states /= [max0-min0, max1-min1]

        self.normalized_states = normalized_states * 2 - 1

    # def plot_esvc(self, path, name, data):
    #     dim = int(np.sqrt(self.env.num_states))
    #     hm = sb.heatmap(np.reshape(data, (dim, dim)))
    #     fig = hm.get_figure()
    #     fig.savefig(path+'/' + name + '.png')
    #     fig.clf()

    # def plot_emp_fc(self, name):
    #     dim = int(np.sqrt(self.env.num_states))
    #     hm = sb.heatmap(np.reshape(self.emp_fc, (dim, dim)).T)
    #     hm.set_title('Empirical Feature Counts')
    #     hm.set_xlabel('x')
    #     hm.set_ylabel('velocity')
    #     fig = hm.get_figure()
    #     fig.savefig(self.output_directory_path + name + '.png')
    #     fig.clf()

    # def plot_esvc_mat(self, path, i):
    #     dim = int(np.sqrt(self.env.num_states))
    #     hm = sb.heatmap(np.reshape(self.esvc_mat[:, -1], (dim, dim)).T)
    #     hm.set_title('Expected State Visitation Counts')
    #     hm.set_xlabel('x')
    #     hm.set_ylabel('velocity')
    #     fig = hm.get_figure()
    #     fig.savefig(path + "esvc_" + str(i) + '.png')
    #     fig.clf()

    def plot_reward(self, nof_iter):
        data = np.reshape(self.state_rewards, self.env.shape)

        hm = sb.heatmap(data)
        fig = hm.get_figure()
        fig.savefig(self.reward_path + str(nof_iter) + '.png')
        fig.clf()

    def plot_in_state_space(self, inp, ind=-1, path=""):
        if path == "":
            path = self.output_directory_path
        else:
            if ind != -1:
                path = path+str(ind)
        data = np.reshape(inp, self.env.shape)
        hm = sb.heatmap(data)
        fig = hm.get_figure()
        fig.savefig(path + '.png')
        fig.clf()

    def save_reward(self, nof_iter):
        self.rewards_file.write(str(nof_iter) + "\n")
        self.rewards_file.write("[")

        for i, r in enumerate(self.state_rewards):
            self.rewards_file.write(str(r))
            if i != len(self.state_rewards)-1:
                self.rewards_file.write(", ")

        self.rewards_file.write("] \n")
        self.rewards_file.flush()

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
        current_s = self.env.find_closest_state(State(s[0], s[1]))
        while not done and step_ctr < 1000:
            env.render()
            action_id = np.random.choice(range(len(self.env.actions)), 1, self.fast_policy[current_s, :].tolist())[0]
            # action_id = np.argmax(self.fast_policy[current_s, :])
            next_s, _, done, _ = env.step(np.array([self.env.actions[action_id].force]))
            current_s = self.env.find_closest_state(State(next_s[0], next_s[1]))
            step_ctr += 1
            # time.sleep(0.01)
            # print("State: ", current_s, " - Action: ", self.env.actions[action_id].force)

        env.close()
