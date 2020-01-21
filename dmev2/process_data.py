"""
  @author: yigit.yildirim@boun.edu.tr
"""

import numpy as np
from mdp_mccont import MCContMDP


def generate_trajectories(mdp):
    trajectories = []
    trajectories_of_ids = []
    for i in range(3):
        demonstrations = np.load(mdp.data_path + 'dems/t_' + str(i) + '.npy')
        j = 0
        for demonstration in demonstrations:
            print('\rTrajectory file: {0}, Demonstration: {1} '.format(i, j), end='')
            j += 1
            trajectory = []
            trajectory_of_ids = []
            for state_action in demonstration:
                sid = mdp.find_closest_state(np.array([state_action[0][0], state_action[0][1]]))
                aid = mdp.find_closest_action(np.array(state_action[1][0]))
                s = (mdp.states[sid]).numpy()
                a = (mdp.actions[aid]).numpy()
                trajectory.append([s, a])
                trajectory_of_ids.append([sid, aid])
            trajectories.append(trajectory)
            trajectories_of_ids.append(trajectory_of_ids)
        print('')
    np.save(mdp.model_path + 'trajectories.npy', np.asarray(trajectories))
    np.save(mdp.model_path + 'trajectories_of_ids.npy', np.asarray(trajectories_of_ids))


if __name__ == "__main__":
    mcc = MCContMDP()
    generate_trajectories(mcc)






