import numpy as np
combined_dict = {}
for i in range(10):
    trajectory_file = f'data/trajectory/trajectories_pregrasp_job_{i}.npz'
    traj_data = np.load(trajectory_file, allow_pickle=True)['trajectories'].item()
    combined_dict.update(traj_data)
np.savez("data/trajectory/trajectories_pregrasp.npz", trajectories=np.array(combined_dict, dtype=object))
