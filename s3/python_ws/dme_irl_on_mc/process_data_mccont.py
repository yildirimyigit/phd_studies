"""
  @author: yigit.yildirim@boun.edu.tr
"""

import numpy as np
from mccont_mdp import MCContMDP


# Loads previously recorded demonstrations to create trajectories
# :param e Environment object
def generate_trajectories(mdp, path='data/mccont/'):
    trajectories = []
    trajectories_of_ids = []
    for i in range(3):
        demonstrations = np.load(path+'t_'+str(i)+'.npy')
        j = 0
        for demonstration in demonstrations:
            print('\rTrajectory file: {0}, Demonstration: {1} '.format(i, j), end='')
            j += 1
            trajectory = []
            trajectory_of_ids = []
            for state_action in demonstration:
                sid = mdp.find_closest_state(np.array([state_action[0][0], state_action[0][1]]))
                aid = mdp.find_closest_action(np.array(state_action[1][0]))
                s = mdp.states[sid]
                a = mdp.actions[aid]
                trajectory.append([s, a])
                trajectory_of_ids.append([sid, aid])
            trajectories.append(np.asarray(trajectory))
            trajectories_of_ids.append(trajectory_of_ids)
        print('')
    np.save(path+'trajectories.npy', np.asarray(trajectories))
    np.save(path+'trajectories_of_ids.npy', np.asarray(trajectories_of_ids))


if __name__ == "__main__":
    mcc = MCContMDP()
    generate_trajectories(mcc)
