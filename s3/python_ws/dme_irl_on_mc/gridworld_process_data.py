"""
  @author: yigit.yildirim@boun.edu.tr
"""

import numpy as np
from gridworld_mdp import GridworldMDP


# Loads previously recorded demonstrations to create trajectories
# :param e Environment object
def generate_trajectories(mdp):
    trajectories = []
    trajectories_of_ids = []

    policy = [[[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 2], [14, 2], [22, 2], [30, 2], [38, 2], [46, 2]],
              [[0, 2], [8, 2], [16, 2], [24, 2], [32, 2], [40, 2], [48, 1], [49, 1], [50, 1], [51, 1], [52, 1], [53, 1]],
              [[0, 1], [1, 2], [9, 1], [10, 2], [18, 1], [19, 2], [27, 1], [28, 2], [36, 1], [37, 2], [45, 1], [46, 2]],
              [[0, 2], [8, 1], [9, 2], [17, 2], [25, 2], [33, 1], [34, 1], [35, 2], [43, 1], [44, 1], [45, 1], [46, 2]]]

    for ind_policy in policy:
        trajectory = []
        trajectory_of_ids = []
        for state_action in ind_policy:
            sid = state_action[0]
            aid = state_action[1]
            s = mdp.states[sid]
            a = mdp.actions[aid]
            trajectory.append([s, a])
            trajectory_of_ids.append([sid, aid])
        trajectories.append(np.asarray(trajectory))
        trajectories_of_ids.append(trajectory_of_ids)
    print('')
    np.save(mdp.env_path+'trajectories.npy', np.asarray(trajectories))
    np.save(mdp.env_path+'trajectories_of_ids.npy', np.asarray(trajectories_of_ids))


if __name__ == "__main__":
    mcc = GridworldMDP()
    generate_trajectories(mcc)
