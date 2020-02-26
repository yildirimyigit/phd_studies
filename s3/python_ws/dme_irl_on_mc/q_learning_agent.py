"""
  @author: yigit.yildirim@boun.edu.tr
"""

import numpy as np
from mccont_mdp import MCContMDP
from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from time import time
import os
import gym


class MCContQLearning:
    def __init__(self):
        self.env = MCContMDP()
        self.episode = 100000
        self.learning_rate = 0.9
        self.gamma = 0.9
        self.epsilon = 0.25  # epsilon-greedy

        self.env_start = self.env.get_start_state()
        self.env_goal = self.env.get_goal_state()

        # when recovering the policy, we use the estimated rewards
        self.is_recovering = True
        self.state_rewards = np.zeros(len(self.env.states))

        # initialization of q matrix
        self.q = np.random.rand(self.env.num_states, self.env.num_actions)/100.0

        self.episode_rewards = np.zeros(self.episode)

        self.gym_env = gym.make('MountainCarContinuous-v0')
        self.gym_env = gym.wrappers.Monitor(self.gym_env, self.env.env_path + "video/" + str(time()),
                                            video_callable=lambda episode_id: False)
        self.gym_env.reset()

    def qlearn(self):
        for i in tqdm(range(self.episode)):
            self.gym_env.reset()
            state = self.env.find_closest_state(self.gym_env.unwrapped.state)
            num_steps, reward, chosen_action, prev_state, done = 0, 0, 0, 0, False

            # finish episode when goal reached or max 10000 steps
            while (not done) and (state not in self.env_goal) and (num_steps < 10000):
                prev_state = state
                chosen_action = self.choose_action(state)
                if np.random.rand() < self.epsilon:  # epsilon-greedy
                    chosen_action = np.random.choice(range(self.env.num_actions))
                state, reward, done = self.act(chosen_action)
                self.episode_rewards[i] += reward
                self.q[prev_state][chosen_action] = self.q[prev_state][chosen_action] + self.learning_rate * \
                    (reward + self.gamma * np.max(self.q[state][:]) - self.q[prev_state][chosen_action])
                num_steps += 1

            if state not in self.env_goal:  # if hit max_steps
                reward = -10000

            self.episode_rewards[i] += reward
            self.q[prev_state][chosen_action] = self.q[prev_state][chosen_action] + self.learning_rate * \
                (reward + self.gamma * np.max(self.q[state][:]) - self.q[prev_state][chosen_action])

    def choose_action(self, state):
        return np.argmax(self.q[state][:])
        # np.random.choice(range(len(self.env.actions)), 1, self.fast_policy[current_s, :].tolist())[0]

    def act(self, action_id):
        next_s, _, done, _ = self.gym_env.step(np.array([self.env.actions[action_id]]))
        s_new = self.env.find_closest_state(next_s)
        r = self.get_reward(s_new)
        return s_new, r, done

    def plot_q(self):
        fonts = [{'family': 'monospace', 'color': 'k', 'weight': 'normal', 'size': 10},
                 {'family': 'monospace', 'color': 'blue', 'weight': 'bold', 'size': 12}]

        fig, ax = plt.subplots(1, 1, figsize=(self.env.v_div * 2, self.env.x_div * 2))
        ax.set_xlim(0, self.env.v_div)
        ax.set_ylim(0, self.env.x_div)
        plt.xticks(np.arange(0.5, self.env.v_div + 0.5, 1), [str(i) for i in range(self.env.v_div)])
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

        q_suffix = 'q_rec.png' if self.is_recovering else 'q.png'

        plt.savefig(self.env.env_path + q_suffix)
        # plt.show(block=True)

    def run(self):
        outdir = self.env.env_path + "video/" + str(time())
        os.makedirs(outdir)
        env = gym.make('MountainCarContinuous-v0')
        env = gym.wrappers.Monitor(env, outdir, video_callable=lambda episode_id: True)
        done = False
        step_ctr = 0

        s = env.reset()
        current_s = self.env.find_closest_state(s)
        while not done and step_ctr < 3500:
            env.render()
            action_id = self.choose_action(current_s)
            next_s, _, done, _ = env.step(np.array([self.env.actions[action_id]]))
            current_s = self.env.find_closest_state(next_s)
            step_ctr += 1

        env.close()

    def save_policy(self):
        q = self.q.copy()
        q[q < 0] = 0
        policy = q / np.reshape(np.sum(q, axis=1), (self.env.num_states, 1))
        np.save(self.env.env_path + 'policy.npy', policy)

    def get_reward(self, s):
        if not self.is_recovering:
            r = 100 if s == self.env_goal else -1
        else:
            r = self.state_rewards[s]

        return r

    def read_reward(self, reward_path):
        file = open(reward_path, 'r')
        lines = file.readlines()
        last_line = lines[-1][1:-3]
        file.close()
        rewards = list(map(float, last_line.split(',')))
        self.state_rewards = np.array(rewards)

    def load_reward(self, reward_path):
        self.state_rewards = np.reshape(np.load(reward_path), np.shape(self.state_rewards))


if __name__ == "__main__":
    mcq = MCContQLearning()
    mcq.read_reward(
        '/home/yigit/Desktop/github/s3/python_ws/dme_irl_on_mc/data/mccont/output/1582272998/reward/rewards.txt')
    mcq.qlearn()
    mcq.run()
    mcq.save_policy()
