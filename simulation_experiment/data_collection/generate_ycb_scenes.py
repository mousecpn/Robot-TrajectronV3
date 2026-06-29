import argparse
from pathlib import Path
import os

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from utils_exp.grasp import Grasp, Label, write_grasp
from utils_exp.visual import grasp2mesh
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections
from utils_exp.implicit import get_mesh_pose_list_from_world, get_scene_from_mesh_pose_list
import hashlib


def main():
    num_scenes = 50
    object_count = 4
    min_object_count = 3

    root = "data/ycb_scene_packed"
    if not os.path.exists(root):
        os.makedirs(root)
    sim = ClutterRemovalSim("packed", "ycb_packed", gui=True)

    for i in range(num_scenes):
        sim.reset(object_count)
        if sim.num_objects < min_object_count:
            print(f"Scene {i} has only {sim.num_objects} objects, skipping.")
            continue
        mesh_pose_list = get_mesh_pose_list_from_world(sim.world, sim.object_set, exclude_plane=True)
        # Generate a unique scene_id based on the mesh_pose_list
        mesh_pose_bytes = str(mesh_pose_list).encode('utf-8')
        scene_id = hashlib.md5(mesh_pose_bytes).hexdigest()
        # write_mesh(Path(root), scene_id, mesh_pose_list)
        # sim.robot.add_robot()
        # sim.save_state()


if __name__=="__main__":

    main()