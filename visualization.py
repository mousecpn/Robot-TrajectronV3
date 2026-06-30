import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import torch
from dataset.se3_preprocessing import load_data, se3_derivatives_of, js_derivatives_of, plot_se3_poses
from model.trajectron import Trajectron
from argument_parser import args
import json
import imageio
from pathlib import Path
from model.transform import SO3_R3, quaternion_to_matrix, Transform, select_grasps
from scipy.spatial.transform import Rotation
from model.io import read_point_cloud
import time

relative = True
save_fig = True

def main():
    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    if args.device.type == 'cuda':
        torch.cuda.set_device(args.device)

    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['frequency'] = 20
    dim = 6

    scene_file = Path("data/data_packed_train_raw")
    trajectory_file = Path("data/trajectory/trajectories_pregrasp_packed.npz")
    dt = 1.0/hyperparams['frequency']

    scene_data, train_traj_data, test_traj_data = load_data(scene_file, trajectory_file, dt, viz=True)

    trajectron = Trajectron(hyperparams, args.device)
    model = torch.load(args.checkpoint, map_location=args.device)
    trajectron.model.node_modules = model
    trajectron.set_annealing_params()
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon'] 
    trajectron.model.to(args.device)
    trajectron.model.eval()
    # trajectron.model.inference_mode()
    # os.makedirs('gif_images', exist_ok=True)
    filenames = []
    T_body_tcp = Transform.from_dict({"rotation": [0.000, 0.000, 0.0, 1.0], "translation": [0.000, 0.000, 0.05]}).as_matrix()
    T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05]).as_matrix()


    scene_l = np.arange(len(test_traj_data))
    np.random.shuffle(scene_l)
    for l in scene_l:
        seq = test_traj_data[l]
        js_traj = torch.tensor(seq['joint_states'],  dtype=torch.float)
        ee_traj_ori = torch.tensor(seq['ee_poses'], dtype=torch.float)
        
        # noise = torch.rand(ee_traj_ori.shape[0], 6) * torch.tensor([0.005, 0.005, 0.005, 0.0025, 0.0025, 0.0025], dtype=torch.float32)
        # noisy_transform = SO3_R3().exp_map(noise).to_matrix()
        # ee_traj_ori = noisy_transform @ ee_traj_ori 

        T_base_task = torch.tensor(seq['robot_base_pose'], dtype=torch.float)
        T_task_base = torch.inverse(T_base_task)
        scene_id = seq['scene_id']
        pcl_ = read_point_cloud(Path("data/scene_packed"), scene_id)
        pcl_ = torch.tensor(pcl_, dtype=torch.float32)
        pcl_ = T_base_task @ torch.cat((pcl_,torch.ones_like(pcl_[:,:1])), dim=-1).unsqueeze(-1)
        
        # grasps
        grasps_data = scene_data[scene_id]
        rotations = quaternion_to_matrix(torch.tensor(grasps_data[:, :4], dtype=torch.float))
        translations = torch.tensor(grasps_data[:, 4:7], dtype=torch.float)
        
        all_points = torch.cat([(T_task_base @ ee_traj_ori)[:, :3, 3], translations], dim=0).cpu().numpy()
        min_xyz = all_points.min(axis=0)
        max_xyz = all_points.max(axis=0)
        padding = 0.02
        x_range = (min_xyz[0] - padding, max_xyz[0] + padding)
        y_range = (min_xyz[1] - padding, max_xyz[1] + padding)
        z_range = (min_xyz[2] - padding, max_xyz[2] + padding)


        ee_vel_traj = se3_derivatives_of(ee_traj_ori, dt=dt)
        # ee_traj_logmap = SO3_R3.from_matrix(ee_traj_ori).log_map()
        js_vel_traj = js_derivatives_of(js_traj, dt=dt)

        # ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1)  # [timestep, 7]
        js_traj = torch.cat((js_traj, js_vel_traj), dim=-1)  # [timestep, 7]
        
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        # ax = plt.axes(projection='3d')
        # plt.xlabel('Y-axis', fontsize=12) 
        # plt.ylabel('X-axis', fontsize=12)
        # ax.set_axis_off()
        # ax.axes.get_xaxis().set_visible(False)
        # ax.axes.get_yaxis().set_visible(False)
        # ax.axes.xaxis.set_ticklabels([])
        # ax.axes.yaxis.set_ticklabels([])
        plt.grid()
        # ax.axes.zaxis.set_ticklabels([])
        # ax.axes.get_zaxis().set_visible(False)
        if not save_fig:
            plt.ion()
        else:
            if 'gif_images' not in os.listdir():
                os.makedirs('gif_images', exist_ok=True)
        # data = data[::2,:]
        steps = ee_traj_ori.shape[0] - 8
        # x_range = (-0.4, 0.6)
        # y_range = (-0.3, 0.8)
        # z_range = (0, 0.5)
        # x_range = (-0.3, 0.3)
        # y_range = (-0.3, 0.3)
        # z_range = (-0.3, 0.3)
        ax.set_xlim([x_range[0], x_range[1]])
        ax.set_ylim([y_range[0], y_range[1]])
        ax.set_zlim([z_range[0], z_range[1]])
        ax.set_box_aspect((x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0]))
        # plt.tick_params(axis='both', labelsize=11)
        # ax.set_aspect('equal', adjustable='datalim')
        curves = None

        for j in range(steps):
            first_history_index = torch.LongTensor(np.array([0])).cuda()
            T_curpose_base = torch.inverse(ee_traj_ori[j+7])
            if relative:
                ee_traj_rel = T_curpose_base @ ee_traj_ori
            else:
                ee_traj_rel = ee_traj_ori
            ee_traj_logmap = SO3_R3.from_matrix(ee_traj_rel).log_map()
            ee_traj = torch.cat((ee_traj_logmap, ee_vel_traj), dim=-1) 

            
            x = ee_traj[j:j+8,:].unsqueeze(0).cuda()
            y = ee_traj[j+8:j+20,:].unsqueeze(0).cuda()
            q = js_traj[j:j+8,:].unsqueeze(0).cuda()
            # ph = data.shape[0]-(j+8)
            
            if relative:
                pcl = T_curpose_base @ pcl_
            else:
                pcl = pcl_
            pcl = pcl[:,:3,0]

            grasps_viz = SO3_R3(rotations, translations).to_matrix()
            grasps_viz = grasps_viz @ torch.tensor(T_body_tcp, dtype=torch.float) @ torch.tensor(T_grasp_pregrasp, dtype=torch.float)
            grasps_viz = select_grasps(grasps_viz, T_task_base @ ee_traj_ori[-1])
            
            grasps = T_base_task @ grasps_viz
            if relative:
                grasps = T_curpose_base @ grasps 
            grasps = select_grasps(grasps)
            grasps = SO3_R3.from_matrix(grasps).log_map()
            
            # rotation_6d = matrix_to_rotation_6d(grasps[:,:3,:3])
            # grasps_data = torch.cat((grasps[:,:3,3], rotation_6d), dim=-1)  # [timestep, 9]

            context = {
                'grasp': [grasps.cuda()],
                'pcl': [pcl.cuda()]
            }


            
            batch = (first_history_index, x, q, y, context)

            
            t1 = time.time()
            with torch.no_grad():
                ################# most likely ##############################
                y_dist, a_dist, predictions = trajectron.predict(batch,
                                        ph=ph,
                                        num_samples=1, # doesn't matter when all_z_sep is true
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=True,
                                        full_dist=False,
                                        dist=True)
            t2 = time.time()
            # batch_ade = evaluation.compute_ade(predictions, y[...,0:dim].cpu()).detach().numpy() * 1000
            # batch_fde = evaluation.compute_fde(predictions, y[...,0:dim].cpu()).detach().numpy() * 1000
            # print("ade {:.2f}, fde {:.2f}".format( batch_ade[0], batch_fde[0]))
            # except:
            #     pass

            if not save_fig:
                trans_entropy_lb = trajectron.model.trans_entropy_lb.cpu().item()
                trans_entropy_ub = trajectron.model.trans_entropy_ub.cpu().item()
                rot_entropy_lb = trajectron.model.rot_entropy_lb.cpu().item()
                rot_entropy_ub = trajectron.model.rot_entropy_ub.cpu().item()
                print("trans entropy lb: {:.4f}, ub: {:.4f}; rot entropy lb: {:.4f}, ub: {:.4f}".format(
                    trans_entropy_lb, trans_entropy_ub, rot_entropy_lb, rot_entropy_ub))

            mode_score = np.exp(trajectron.model.latent.p_dist.logits.detach().cpu().numpy()[0,0])
            
            # ax.plot(vis_data[:,1], vis_data[ :,0], '#34638D')
            # ax.scatter(vis_data[::2,1], vis_data[::2,0], s=5, c='#34638D')

            if curves is not None:
                if type(curves) == list:
                    for c in curves:
                        c.remove()
                else:
                    curves.remove()
            
            vis_data = ee_traj_ori[j:j+8]
            if j == 0:
                plot_se3_poses(T_task_base@vis_data, other_poses=grasps_viz, ax=ax, color='green', draw_frame_flag=False)
            else:
                plot_se3_poses(T_task_base@vis_data, ax=ax, color='green', axis_scale=0.01, alpha=0.5, draw_frame_flag=False)
            
            # curve, = ax.plot(vis_pred[0, :,0], vis_pred[0, :,1], 'red')
            curves = []
            predictions = np.concatenate((np.eye(4).reshape(1,1,1,4,4).repeat(predictions.shape[0], axis=0), predictions), axis=2)
            for s in range(predictions.shape[0]):
                if relative:
                    pred_traj_viz = plot_se3_poses(T_task_base@ee_traj_ori[j+7]@predictions[s].reshape(-1,4,4), ax=ax, lineform='--', alpha=mode_score[s]*0.9+0.1, color='red')
                else:
                    pred_traj_viz = plot_se3_poses(T_task_base@predictions[s].reshape(-1,4,4), ax=ax, lineform='--', alpha=mode_score[s]*0.9+0.1, color='red')
                curves.extend(pred_traj_viz)
                # curve.append(ax.plot(vis_pred[0, :,1], vis_pred[0, :,0], '#34638D', linestyle='--', alpha=mode_score[s] ))  
                
                # curve.append(ax.plot(vis_pred[0, :,0], vis_pred[0, :,1], 'red')) 
            if save_fig:
                if j % 4 == 0:
                    img_file_name = 'gif_images/traj_index{}_step{}.png'.format(l,j)
                    filenames.append(img_file_name)
                    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
                    plt.savefig(img_file_name, dpi = 300) # bbox_inches='tight', pad_inches=0.1
            else:
                plt.pause(0.1)
                plt.ioff()

        # # data = data
        # ax.plot(data[:,1], data[ :,0], 'blue')
        # ax.scatter(data[::2,1], data[::2,0], s=5, c='green')
        # # plt.show()
        # plt.pause(0.02)
        plt.close(fig)

        if save_fig:
            with imageio.get_writer("traj{}.mp4".format(l),fps=10) as writer:
                for filename in filenames:
                    writer.append_data(imageio.v2.imread(filename))
            print("Saved traj{}.mp4".format(l))
            filenames = []
        
        # ax.set_title('3D line plot')
        # plt.show()
    return


if __name__=="__main__":
    main()