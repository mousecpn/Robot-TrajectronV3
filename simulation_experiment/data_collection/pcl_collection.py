import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import os
import sys

# Add parent directory to path (Robot-TrajectronV3)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils_exp.grasp import Grasp, Label
from utils_exp.io import *
from utils_exp.perception import *
from utils_exp.visual import grasp2mesh
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections
from utils_exp.implicit import get_scene_from_mesh_pose_list
from utils_exp.visual import trimesh_to_open3d, visualize_point_cloud_with_normals
State = collections.namedtuple("State", ["tsdf", "pc"])

debug = True


def read_mesh(root, scene_id, name="mesh_pose_list"):
    path = root / name / (scene_id + ".npz")
    return np.load(path, allow_pickle=True)['pc']

def main(args):
    global debug
    root = "data/data_scene_raw"
    save_dir = "data/scene_data"
    df = read_df(Path(root))
    scene_list = list(set(df.loc[:,"scene_id"].to_numpy()))
    scene_list.sort()
    sim = ClutterRemovalSim("pile", "pile/train", gui=debug, save_file_name=args.save_file_name, randomview=True, add_noise='dex')

    Path('/'.join([save_dir,"point_clouds"])).mkdir(parents=True, exist_ok=True)
    Path('/'.join([save_dir,"tsdf"])).mkdir(parents=True, exist_ok=True)

    traj_count = 0
    cur_idx = args.start_scene
    print("start index:", cur_idx)
    while cur_idx < args.start_scene + (args.num_scenes if (args.num_scenes is not None and args.start_scene + args.num_scenes < len(scene_list)) else len(scene_list)):
        scene_id = scene_list[cur_idx]
        mesh_pose_list = read_mesh(Path(root), scene_id)
        sim.recovered_scene(mesh_pose_list)
        sim.save_state()
        # pc = read_point_cloud(Path("data/scene_pile"), scene_id)

        # pcl = read_point_cloud(Path("data/scene_packed"), scene_id)

        sim.robot.add_robot()

        tsdf, pc, timing = sim.acquire_tsdf(1, 1, 40)
        pc = np.asarray(pc.points)
        # visualize_point_cloud_with_normals(pc)

        write_point_cloud(Path(save_dir), scene_id, pc)
        cur_idx += 1



    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with save interval and directory as arguments")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Interval (in steps) between saves")
    parser.add_argument("--save_file_name", type=str, default="./data/trajectory",
                        help="Directory where checkpoints will be saved")
    parser.add_argument("--no-debug", action='store_true', 
                        help="Enable debug mode for additional output")
    parser.add_argument("--start-scene", type=int, default=0, 
                        help="Start from a specific scene index")
    parser.add_argument("--num-scenes", type=int, default=None,
                        help="Number of scenes to process, if None, process all")
                    
    args = parser.parse_args()
    if args.no_debug:
        debug = False
        import logging
        logging.basicConfig(level=logging.ERROR)

    main(args)