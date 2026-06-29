import argparse
from pathlib import Path
import os
import sys

# Add parent directory to path (Robot-TrajectronV3)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
import numpy as np

from utils_exp import Grasp, Label, read_df, read_mesh, write_grasp
from experiment.simulation import ClutterRemovalSim
from utils_exp.transform import Rotation, Transform
import collections

State = collections.namedtuple("State", ["tsdf", "pc"])
twist_msg = np.zeros((7))

gripper_adjusted = Transform(Rotation.from_euler("z", np.pi/2), [0,0,0])
T_tcp_body = Transform(Rotation.identity(), [0,0, 0.022])



def load_grasp_set(world):
    """
    Load precomputed grasps for YCB Objects.
    returns a dictionary with body uid as key and a list of transforms as value.
    """
    precomputed_grasps = dict()
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' or name == 'panda':
            continue
        body = world.bodies[uid]
        pose = body.get_pose().as_matrix()
        scale = body.scale
        visuals = world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, scale, mesh_path, _, _, _ = visuals[0]
        scale = scale[0] # assuming uniform scaling
        mesh_path = mesh_path.decode('utf8')
        obj_name = mesh_path.split('/')[-2]
        grasp_path = "data/ycb_grasps/simulated/{}.npy".format(obj_name)
        if not os.path.exists(grasp_path):
            print("Simulator generated poses for {} not found!".format(obj_name))
            continue
        try:
            simulator_grasp = np.load(grasp_path, allow_pickle=True)
            pose_grasp = simulator_grasp.item()["transforms"]
        except:
            simulator_grasp = np.load(
                grasp_path,
                allow_pickle=True,
                fix_imports=True,
                encoding="bytes",
            )
            pose_grasp = simulator_grasp.item()[b"transforms"]
        # apply scaling
        pose_grasp[:, :3, 3] *= scale
        precomputed_grasps[uid] = pose_grasp

    return precomputed_grasps

def add_triad(p, bodyId, transform: Transform, length=0.05, width=0.1):
    trans = transform.translation
    rot = transform.rotation.as_matrix()
    tips = trans[:, None] + length * rot
    tips = tips.T
    # x axis
    p.addUserDebugLine(lineFromXYZ=trans,
                       lineToXYZ=tips[0],
                       lineColorRGB=[1,0,0],
                       lineWidth=width,
                       parentObjectUniqueId=bodyId)
    # y axis
    p.addUserDebugLine(lineFromXYZ=trans,
                       lineToXYZ=tips[1],
                       lineColorRGB=[0,1,0],
                       lineWidth=width,
                       parentObjectUniqueId=bodyId)
    # z axis
    p.addUserDebugLine(lineFromXYZ=trans,
                       lineToXYZ=tips[2],
                       lineColorRGB=[0,0,1],
                       lineWidth=width,
                       parentObjectUniqueId=bodyId)
    return



def main():
    global twist_msg
    scale = 0.8
    root = Path("data/ycb_scene_packed")
    scene_list = [p.stem for p in root.rglob("*.npz")]
    sim = ClutterRemovalSim("packed", "ycb_packed", gui=False)


    for i in range(len(scene_list)):
        scene_id = scene_list[i]
        mesh_pose_list = read_mesh(root, scene_id)
        sim.recovered_scene(mesh_pose_list)
        # sim.robot.add_robot()
        sim.save_state()

        """
        df = read_df(Path(root))
        scene_mask = df.loc[:, "scene_id"] == scene_id
        label_array = df.loc[:, "label"].to_numpy(np.single)
        pos_mask = (label_array>0.4)
        mask = scene_mask & pos_mask
        scene_data = df.loc[mask, "qx":"label"].to_numpy(np.single)
        grasp_mesh_list = []
        for j in range(len(scene_data)):
            label = scene_data[j,-1]
            ori = scene_data[j, :4]
            pos = scene_data[j, 4:7]
            g = Grasp(Transform(Rotation.from_quat(ori), pos), 0.08)
            grasp_mesh_list.append(trimesh_to_open3d(grasp2mesh(g, label)))

        scene = trimesh_to_open3d(get_scene_from_mesh_pose_list(mesh_pose_list, return_list=False))
        o3d.visualization.draw_geometries([scene, ]+ grasp_mesh_list,
                                            window_name="Point Cloud with Normals",
                                            point_show_normal=True)
        

        # for i in range(len(scene_data)):
        #     grasp = Transform.from_list(scene_data[i,:7])
        #     g = Grasp(grasp, 0.08)
        #     result, width = sim.execute_grasp(g, remove=False, allow_contact=False)
        #     sim.restore_state()
        print()
        """

        
        # load object-centric grasps
        grasps = load_grasp_set(sim.world)
        # visualize loaded grasps
        for uid, transforms in grasps.items():
            object_pos, object_ori = sim.world.p.getBasePositionAndOrientation(uid)
            object_pos = np.array(object_pos) #* scale
            object_pose = Transform.from_list(list(object_ori)+list(object_pos))
            for transform in transforms:
                grasp_pose = object_pose * Transform.from_matrix(transform) * T_tcp_body * gripper_adjusted
                g = Grasp(grasp_pose, 0.08)
                translation = grasp_pose.translation
                # print("translation:", translation)
                if translation[0] > 0.28 or translation[0] < 0.02 or translation[1] > 0.28 or translation[1] < 0.02 or translation[2] > 0.28 or translation[2] < 0.055:
                    # print("invalid grasp")
                    continue
                result, width = sim.execute_grasp(g, remove=False, allow_contact=False)
                g.width = width
                sim.restore_state()
                if result == 1:
                    write_grasp(root, scene_id, g, 1)

                # add_triad(sim.world.p, uid, Transform.from_matrix(transform))

        # print("Loaded {} grasps for scene {}".format(sum([len(grasp[1]) for grasp in grasps.items()]), scene_id))
        

    return

if __name__=="__main__":

    main()