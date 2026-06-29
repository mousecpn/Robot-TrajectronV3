import numpy as np
def write_point_cloud(root, scene_id, point_cloud, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    np.savez_compressed(path, pc=point_cloud)

def read_point_cloud(root, scene_id, name="point_clouds"):
    path = root / name / (scene_id + ".npz")
    return np.load(path)["pc"]
