"""
  @author: yigit.yildirim@boun.edu.tr

  - Wulfmeier 2016, MaxEnt Deep IRL
  - Kitani 2012, Activity Forecasting
"""

import numpy as np
from agent import IRLAgent


class DME:
    def __init__(self):
        self.irl_agent = IRLAgent()
        self.iter_count = 10000

        self.losses = np.zeros((self.iter_count, len(self.irl_agent.emp_fc)))
        self.euler_losses = np.zeros(self.iter_count)

    def run(self):
        state_array = np.asarray(self.irl_agent.env.state_list)

        for i in range(self.iter_count):
            print('--- Iteration {0} ---'.format(i))
            # calculate state rewards
            self.irl_agent.state_rewards = self.irl_agent.reward_batch()

            # solve mdp wrt current reward
            self.irl_agent.backward_pass()
            self.irl_agent.forward_pass()   # calculate irl.esvc to use it in calculation of irl.exp_fc

            # calculate loss and euler distance to [0,0, ..., 0] which we want loss to be
            loss = self.irl_agent.emp_fc - self.irl_agent.exp_fc()
            euler_loss = np.power(np.sum(np.power(loss, 2)), 0.5)

            self.irl_agent.rew_nn.backprop_diff(euler_loss, state_array, self.irl_agent.state_rewards)

            print("Loss: "+str(euler_loss))
            self.losses[i] = loss
            self.euler_losses[i] = euler_loss


if __name__ == "__main__":
    dme = DME()
    dme.run()
