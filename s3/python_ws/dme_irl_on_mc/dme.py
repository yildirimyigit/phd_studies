"""
  @author: yigit.yildirim@boun.edu.tr

  - Wulfmeier 2016, MaxEnt Deep IRL
  - Kitani 2012, Activity Forecasting
"""

import numpy as np
from agent import IRLAgent

import seaborn as sb

import os
import time
import matplotlib.pyplot as plt


class DME:
    def __init__(self):
        self.irl_agent = IRLAgent()
        self.iter_count = 10000

        # self.losses = np.zeros((self.iter_count, len(self.irl_agent.emp_fc)))
        self.cumulative_dists = np.zeros(self.iter_count)

        # #######################################################################
        # create the directory to be used for plotting for rewards
        self.reward_path = self.irl_agent.output_directory_path + 'reward/'
        self.loss_path = self.irl_agent.output_directory_path + 'loss/'
        os.makedirs(self.reward_path)
        os.makedirs(self.loss_path)
        # #######################################################################
        self.rewards_file = open(self.reward_path + 'rewards.txt', "a+")
        # self.rewards_file0 = open(self.reward_path + 'rewards0.txt', "a+")
        # #######################################################################

    def run(self):

        lr = 1e-3
        decay = 1e-7

        for i in range(self.iter_count):
            print('--- Iteration {0} ---'.format(i))
            # calculate state rewards

            self.irl_agent.reward_batch()

            self.save_reward(i)
            self.plot_reward(i)
            # self.plot_reward2(i)

            # solve mdp wrt current reward
            t0 = time.time()
            self.irl_agent.fast_backward_pass()
            t1 = time.time()
            self.irl_agent.fast_forward_pass()   # calculate irl.esvc to use it in calculation of irl.exp_fc
            t2 = time.time()
            print('Duration-- back: {0}, forward: {1}'.format(t1-t0, t2-t1))

            # calculate loss and euler distance to [0,0, ..., 0] which we want loss to be
            # loss = self.irl_agent.emp_fc - self.irl_agent.exp_fc()  # FAULTY exp_fc calculation
            diff = self.irl_agent.emp_fc - self.irl_agent.esvc
            dist = np.power(diff, 2)

            lr = np.maximum(lr - decay, 1e-10)
            self.irl_agent.backpropagation_batch(dist, lr)

            self.cumulative_dists[i] = np.sum(dist) * 1e6
            print("Distance:" + str(self.cumulative_dists[i])+"\n")
            self.plot_cumulative_dists(i)

    def plot_reward(self, nof_iter):
        dim = int(np.sqrt(len(self.irl_agent.env.state_list)))
        data = np.reshape(self.irl_agent.state_rewards, (dim, dim))

        hm = sb.heatmap(data)
        fig = hm.get_figure()
        fig.savefig(self.reward_path + str(nof_iter) + '.png')
        fig.clf()

    def save_reward0(self, nof_iter):
        self.rewards_file0.write(str(nof_iter) + "\n")
        self.rewards_file0.write("[")

        for i, r in enumerate(self.irl_agent.state_rewards):
            self.rewards_file0.write(str(r))
            if i != len(self.irl_agent.state_rewards)-1:
                self.rewards_file0.write(", ")

        self.rewards_file0.write("] \n")
        self.rewards_file0.flush()

    def save_reward(self, nof_iter):
        self.rewards_file.write(str(nof_iter) + "\n")
        self.rewards_file.write("[")

        for i, r in enumerate(self.irl_agent.state_rewards):
            self.rewards_file.write(str(r))
            if i != len(self.irl_agent.state_rewards)-1:
                self.rewards_file.write(", ")

        self.rewards_file.write("] \n")
        self.rewards_file.flush()

    def plot_reward2(self, nof_iter):
        # plt.ylim(-0.2, 0.2)
        plt.plot(range(len(self.irl_agent.state_rewards)), self.irl_agent.state_rewards)
        plt.savefig(self.reward_path + '_' + str(nof_iter) + '.png')
        plt.clf()

    def plot_reward_delta(self, delta, i):
        dim = int(np.sqrt(len(self.irl_agent.env.state_list)))
        data = np.reshape(delta, (dim, dim))

        hm = sb.heatmap(data)
        fig = hm.get_figure()
        fig.savefig(self.reward_path + 'delta_' + str(i) + '.png')
        fig.clf()

    def plot_cumulative_dists(self, i):
        plt.plot(range(i), self.cumulative_dists[:i])
        plt.savefig(self.loss_path + str(i) + '.png')
        plt.clf()

    # testing value iteration algorithm of the agent
    def test_backward_pass(self):
        self.irl_agent.state_rewards = -np.ones((len(self.irl_agent.env.states), 1))
        self.irl_agent.state_rewards[self.irl_agent.env.goal_id] = 100

        # self.irl_agent.fast_backward_pass()
        # draw_advantage(self.irl_agent.advantage)


def draw_advantage(p):
    td = np.max(p, axis=1)
    dim = int(np.sqrt(len(td)))
    data = np.reshape(td, (dim, dim))
    _ = sb.heatmap(np.exp(data))
    plt.show()


if __name__ == "__main__":
    dme = DME()
    dme.run()
    # dme.test_backward_pass()
