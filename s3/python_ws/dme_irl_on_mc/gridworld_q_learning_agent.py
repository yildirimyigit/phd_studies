"""
  @author: yigit.yildirim@boun.edu.tr
"""

import numpy as np
from gridworld_mdp import GridworldMDP
from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches


class GridworldQLearning:
    def __init__(self):
        self.env = GridworldMDP()
        self.episode = 20000
        self.learning_rate = 0.9
        self.gamma = 0.9
        self.epsilon = 0.25  # epsilon-greedy

        self.env_start = self.env.get_start_state()
        self.env_goal = self.env.get_goal_state()

        # initialization of q matrix
        self.q = np.random.rand(self.env.num_states, self.env.num_actions)/100.0

        self.episode_rewards = np.zeros(self.episode)

    def qlearn(self):
        for i in tqdm(range(self.episode)):
            state = self.env_start
            num_steps, reward, chosen_action, prev_state = 0, 0, 0, 0

            while (state != self.env_goal) and (num_steps < 160):  # finish episode when goal reached or max 160 steps
                prev_state = state
                chosen_action = self.choose_action(state)
                if np.random.rand() < self.epsilon:  # epsilon-greedy
                    chosen_action = np.random.choice(range(self.env.num_actions))
                state, reward = self.act(state, chosen_action)
                self.episode_rewards[i] += reward
                self.q[prev_state][chosen_action] = self.q[prev_state][chosen_action] + self.learning_rate * \
                    (reward + self.gamma * np.max(self.q[state][:]) - self.q[prev_state][chosen_action])
                num_steps += 1

            if state != self.env_goal:  # if hit max_steps
                reward = -10000

            self.episode_rewards[i] += reward
            self.q[prev_state][chosen_action] = self.q[prev_state][chosen_action] + self.learning_rate * \
                (reward + self.gamma * np.max(self.q[state][:]) - self.q[prev_state][chosen_action])

    def choose_action(self, state):
        return np.argmax(self.q[state][:])

    def act(self, s, a):
        s_new = np.where(self.env.transitions[s, a, :] == 1)[0][0]
        r = 100 if s_new == self.env_goal else -1
        return s_new, r

    def plot_q(self):
        fonts = [{'family': 'monospace', 'color': 'k', 'weight': 'normal', 'size': 10},
                 {'family': 'monospace', 'color': 'blue', 'weight': 'bold', 'size': 12}]

        fig, ax = plt.subplots(1, 1, figsize=(self.env.y_div * 2, self.env.x_div * 2))
        ax.set_xlim(0, self.env.y_div)
        ax.set_ylim(0, self.env.x_div)
        plt.xticks(np.arange(0.5, self.env.y_div + 0.5, 1), [str(i) for i in range(self.env.y_div)])
        plt.yticks(np.arange(self.env.x_div - 0.5, -0.5, -1), [str(i) for i in range(self.env.x_div)])

        goal_x, goal_y = self.env.get_coord(self.env_goal)
        ax.add_patch(patches.Rectangle((goal_x, self.env.y_div - goal_y - 1), 1, 1, color="darkred"))
        ax.add_patch(patches.Rectangle((0, self.env.y_div - 1), 1, 1, color="lightgreen"))  # 0, 0 starting cell
        for state in self.env.pits:
            coord = self.env.get_coord(state)
            ax.add_patch(patches.Rectangle(coord, 1, 1, color="gray"))

        for i in range(self.env.x_div):
            for j in range(self.env.y_div):
                ax.add_line(mlines.Line2D([i, i + 1], [j, j + 1], color="darkred"))
                ax.add_line(mlines.Line2D([i, i + 1], [j + 1, j], color="darkred"))

        for i in range(self.env.x_div):
            ax.add_line(mlines.Line2D([i, i], [0, self.env.y_div], color="k"))
        for i in range(self.env.y_div):
            ax.add_line(mlines.Line2D([0, self.env.x_div], [i, i], color="k"))

        offset = [[-0.65, -0.20], [-0.35, -0.55], [-0.65, -0.85], [-0.95, -0.55]]

        for j in range(self.env.y_div):
            for i in range(self.env.x_div):
                if i != goal_x or j != goal_y:
                    s = j * self.env.x_div + i
                    for a in range(self.env.num_actions):
                        if a in np.argwhere(self.q[s, :] == np.amax(self.q[s, :])):
                            ax.text(i + 1 + offset[a][0], self.env.y_div - j + offset[a][1], '%.1f' % self.q[s, a],
                                    fontdict=fonts[1])
                        else:
                            ax.text(i + 1 + offset[a][0], self.env.y_div - j + offset[a][1], '%.1f' % self.q[s, a],
                                    fontdict=fonts[0])

        plt.savefig(self.env.env_path + 'q.png')
        # plt.show(block=True)

    def save_policy(self):
        policy = self.q / np.reshape(np.sum(self.q, axis=1), (self.env.num_states, 1))
        np.save(self.env.env_path + 'policy.npy', policy)


if __name__ == "__main__":
    gq = GridworldQLearning()
    gq.qlearn()
    gq.plot_q()
    gq.save_policy()
