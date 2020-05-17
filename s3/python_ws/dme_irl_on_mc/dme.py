"""
  @author: yigit.yildirim@boun.edu.tr

  - Wulfmeier 2016, MaxEnt Deep IRL
  - Kitani 2012, Activity Forecasting
"""

import numpy as np
import cv2
from agent import IRLAgent

import time
import matplotlib.pyplot as plt


class DME:
    def __init__(self):
        self.irl_agent = IRLAgent()
        self.iter_count = 20000

        # self.losses = np.zeros((self.iter_count, len(self.irl_agent.emp_fc)))
        self.cumulative_dists = np.zeros(self.iter_count)

    def run(self):

        lr = 5e-2
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
            self.irl_agent.new_backward_pass()
            t1 = time.time()
            self.irl_agent.new_forward_pass()   # calculate irl.esvc to use it in calculation of irl.exp_fc
            t2 = time.time()
            print(f'Duration-- back: {t1-t0}, forward: {t2-t1}')

            # self.save_policy(i)

            diff = self.irl_agent.emp_fc - self.irl_agent.esvc
            print("Diff sum: ", repr(np.sum(np.abs(diff))))

            # wssd: wasserstein distance, flow: matrix of individual displacements
            wssd, _, flow = cv2.EMD(esvc_to_sig(np.reshape(self.irl_agent.esvc, self.irl_agent.env.shape)),
                                    esvc_to_sig(np.reshape(self.irl_agent.emp_fc, self.irl_agent.env.shape)),
                                    cv2.DIST_L2)

            # dist = np.abs(diff)
            dist = flow_to_dist_arr(wssd, flow)

            lr = np.maximum(lr - decay, 1e-10)
            self.irl_agent.backpropagation_batch(dist, lr)

            self.cumulative_dists[i] = np.sum(np.abs(dist))
            print("Distance:" + str(self.cumulative_dists[i])+"\n")
            self.plot_cumulative_dists(i)
            # self.irl_agent.plot_esvc_mat(self.irl_agent.esvc_path, i)
            # self.irl_agent.plot_in_state_space(self.irl_agent.esvc, i, self.irl_agent.esvc_path,
            #                                    title='Expected State Visitation Counts')

            self.irl_agent.plot_in_state_space(self.irl_agent.esvc, ind=i, path=self.irl_agent.esvc_path,
                                               title='Empirical Feature Counts')

            # self.save_esvc(i)

            if i % 100 == 0:
                try:
                    self.irl_agent.run_policy(str(i))
                except:
                    pass

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

    def plot_cumulative_dists(self, i):
        plt.plot(range(i), self.cumulative_dists[:i])
        plt.savefig(self.irl_agent.output_directory_path + 'loss.png')
        plt.clf()
        plt.close()

        if i > 0:
            mx = np.max([0, i-100])
            plt.plot(range(mx, i), self.cumulative_dists[mx:i])
            plt.savefig(self.irl_agent.output_directory_path + 'last_100_loss.png')
            plt.clf()
            plt.close()

    # testing value iteration algorithm of the agent
    # def test_backward_pass(self):
    #     self.irl_agent.state_rewards = -np.ones((len(self.irl_agent.env.states), 1))
    #     self.irl_agent.state_rewards[self.irl_agent.env.goal_id] = 100
    #
    #     self.irl_agent.fast_backward_pass()
    #     self.irl_agent.run_policy('0')


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


# to be used in earth mover's (wasserstein) distance calculation
def esvc_to_sig(esvc):
    # Convert esvc to a signature for cv2.EMD
    # cv2.EMD requires single-precision, floating-point input
    sig = np.zeros((np.size(esvc), np.ndim(esvc)+1), dtype=np.float32)

    row = 0
    for i, x in np.ndenumerate(esvc):
        sig[row, 0] = x
        sig[row, 1:] = np.asarray(i)
        row += 1

    return sig


def flow_to_dist_arr(dist, flow):
    num_states = len(flow)
    loss = np.zeros(num_states)
    for i in range(num_states):
        loss[i] = dist * (np.sum(flow[:, i]) - np.sum(flow[i, :]))

    return loss


if __name__ == "__main__":
    dme = DME()
    dme.run()
    # dme.test_backward_pass()
