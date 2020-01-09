"""
  @author: yigit.yildirim@boun.edu.tr

  - Wulfmeier 2016, MaxEnt Deep IRL
  - Kitani 2012, Activity Forecasting
"""

import numpy as np
from agent import IRLAgent

import seaborn as sb

import time
import matplotlib.pyplot as plt


class DME:
    def __init__(self):
        self.irl_agent = IRLAgent()
        self.iter_count = 10000

        # self.losses = np.zeros((self.iter_count, len(self.irl_agent.emp_fc)))
        self.cumulative_dists = np.zeros(self.iter_count)

    def run(self):

        lr = 1e-3
        decay = 1e-8

        for i in range(self.iter_count):
            print('--- Iteration {0} ---'.format(i))
            # calculate state rewards

            self.irl_agent.reward_batch()

            self.irl_agent.save_reward(i)
            self.irl_agent.plot_reward(i)
            # self.plot_reward2(i)

            # solve mdp wrt current reward
            t0 = time.time()
            self.irl_agent.fast_backward_pass()
            t1 = time.time()
            self.irl_agent.fast_forward_pass()   # calculate irl.esvc to use it in calculation of irl.exp_fc
            t2 = time.time()
            print('Duration-- back: {0}, forward: {1}'.format(t1-t0, t2-t1))

            # self.save_policy(i)

            # calculate loss and euler distance to [0,0, ..., 0] which we want loss to be
            # loss = self.irl_agent.emp_fc - self.irl_agent.exp_fc()  # FAULTY exp_fc calculation
            diff = self.irl_agent.emp_fc - self.irl_agent.esvc
            print("Diff sum: ", repr(np.sum(np.abs(diff))))
            # dist = np.power(diff, 2)
            dist = np.pow(diff, 2)

            lr = np.maximum(lr - decay, 1e-10)
            self.irl_agent.backpropagation_batch(dist, lr)

            self.cumulative_dists[i] = np.sum(dist)
            print("Distance:" + str(self.cumulative_dists[i])+"\n")
            self.plot_cumulative_dists(i)
            # self.irl_agent.plot_esvc_mat(self.irl_agent.esvc_path, i)
            self.irl_agent.plot_in_state_space(self.irl_agent.esvc, i, self.irl_agent.esvc_path,
                                               title='Expected State Visitation Counts')
            # self.save_esvc(i)

            if i % 100 == 0:
                try:
                    self.irl_agent.run_policy(str(i))
                except:
                    pass

    # def save_esvc(self, nof_iter):
    #     self.esvc_file.write(str(nof_iter) + "\n")
    #     self.esvc_file.write("[")
    #
    #     for i, r in enumerate(self.irl_agent.esvc_mat[:, -1]):
    #         self.esvc_file.write(str(r))
    #         if i != len(self.irl_agent.esvc_mat[:, -1]) - 1:
    #             self.esvc_file.write(", ")
    #
    #     self.esvc_file.write("] \n")
    #     self.esvc_file.flush()
    #
    # def save_policy(self, ind):
    #     self.policy_file.write(str(ind) + "\n")
    #     self.policy_file.write("[")
    #
    #     for i in range(len(self.irl_agent.env.states)):
    #         self.policy_file.write("[")
    #         for j in range(len(self.irl_agent.env.actions)):
    #             self.policy_file.write(str(self.irl_agent.fast_policy[i, j]))
    #             if j != len(self.irl_agent.env.actions) - 1:
    #                 self.policy_file.write(", ")
    #         self.policy_file.write("]")
    #         if i != len(self.irl_agent.env.states) - 1:
    #             self.policy_file.write(", ")
    #
    #     self.policy_file.write("] \n\n")
    #     self.policy_file.flush()

    # def plot_reward2(self, nof_iter):
    #     plt.plot(range(len(self.irl_agent.state_rewards)), self.irl_agent.state_rewards)
    #     plt.savefig(self.reward_path + '_' + str(nof_iter) + '.png')
    #     plt.clf()
    #
    # def plot_reward_delta(self, delta, i):
    #     dim = int(np.sqrt(len(self.irl_agent.env.state_list)))
    #     data = np.reshape(delta, (dim, dim))
    #
    #     hm = sb.heatmap(data)
    #     fig = hm.get_figure()
    #     fig.savefig(self.reward_path + 'delta_' + str(i) + '.png')
    #     fig.clf()

    def plot_cumulative_dists(self, i):
        plt.plot(range(i), self.cumulative_dists[:i])
        plt.savefig(self.irl_agent.output_directory_path + 'loss.png')
        plt.clf()

    # testing value iteration algorithm of the agent
    # def test_backward_pass(self):
    #     self.irl_agent.state_rewards = -np.ones((len(self.irl_agent.env.states), 1))
    #     self.irl_agent.state_rewards[self.irl_agent.env.goal_id] = 100
    #
    #     self.irl_agent.fast_backward_pass()
    #     self.irl_agent.run_policy('0')


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


if __name__ == "__main__":
    dme = DME()
    dme.run()
    # dme.test_backward_pass()
