# eval.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json
import argparse
from model.trajectron import Trajectron
from dataset.se3_preprocessing import SE3GraspTrajDataset, load_data
import evaluation
from tqdm import tqdm
import os

def evaluate_model(trajectron, eval_dataloader, hyperparams, args, dim=6):
    """
    Run evaluation loop for the model.
    """
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()

    with torch.no_grad():
        eval_loss_list = []
        pbar = tqdm(eval_dataloader, ncols=80)
        ade, fde = [], []

        for batch in pbar:
            (first_history_index, x, q, y, context) = batch
            eval_loss = trajectron.eval_loss(batch)
            pbar.set_description(f"Eval L: {eval_loss.item():.2f}")
            eval_loss_list.append({'nll': [eval_loss]})

            # Predictions
            predictions = trajectron.predict(
                batch,
                ph,
                num_samples=1,
                z_mode=True,
                gmm_mode=True,
                all_z_sep=False,
                full_dist=False
            )
            # fig = plt.figure(figsize=(8,8))
            # ax = fig.add_subplot(111, projection='3d')
            # past_traj = SO3_R3().exp_map(x[0,:,:6]).to_matrix().cpu().numpy()
            # future_traj = SO3_R3().exp_map(y[0,:,:6]).to_matrix().cpu().numpy()
            # grasp_viz = SO3_R3().exp_map(context['grasp'][0][:,:6]).to_matrix().cpu().numpy()
            # plot_se3_poses(np.concatenate((past_traj,future_traj),axis=0), other_poses=grasp_viz, ax=ax, color='green')
            # for i in range(len(predictions)):
            #     plot_se3_poses(predictions[i,0], other_poses=None, ax=ax, color='red')
                
            # plt.show()
            # plt.close(fig)
            # Metrics
            batch_ade = evaluation.compute_ade(predictions, y[..., 0:dim]).detach().cpu().numpy()
            batch_fde = evaluation.compute_fde(predictions, y[..., 0:dim]).detach().cpu().numpy()
            ade.append(batch_ade)
            fde.append(batch_fde)

        ade = np.mean(np.concatenate(ade, axis=0)) * 1000
        fde = np.mean(np.concatenate(fde, axis=0)) * 1000

    print("Final Evaluation Results:")
    print("ADE:", ade)
    print("FDE:", fde)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default='config/config_test.json', help="Path to config JSON")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/epoch10|20Hz|ade21.05.pth', help="Path to trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for evaluation")
    args = parser.parse_args()

    if not torch.cuda.is_available() or args.device == "cpu":
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)
        torch.cuda.set_device(args.device)

    # Load hyperparameters
    if not os.path.exists(args.conf):
        raise FileNotFoundError(f"Config json not found: {args.conf}")
    with open(args.conf, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)

    hyperparams["batch_size"] = args.batch_size
    hyperparams["frequency"] = 20  # 固定 frequency（和训练一致）

    # Load dataset
    scene_files = [Path("data/data_pile_train_fix_raw_graspnet1b"),Path("data/data_packed_train_raw"), ] # Path("data/data_packed_train_raw")
    trajectory_files = [Path("data/trajectory/trajectories_pregrasp_pile2.npz"), Path("data/trajectory/trajectories_pregrasp_zflip.npz"), ] #   Path("data/trajectory/trajectories_pregrasp_zflip_mpc.npz")
    pcl_roots = [Path("data/scene_pile_graspnet1b"), Path("data/scene_packed"), ] #  Path("data/scene_data")
    
    # scene_files = [Path("data/data_packed_train_raw")]
    # trajectory_files = [Path("data/trajectory/trajectories_pregrasp_packed.npz")]
    # pcl_roots = [Path("data/scene_packed")]
    
    dt = 0.05
    relative = True
    concat_test_data = []
    for scene_file, trajectory_file, pcl_root in zip(scene_files, trajectory_files, pcl_roots):
        if 'mpc' in str(trajectory_file):
            dt_term = 0.1
            frequency_term = 1/dt_term
        else:
            dt_term = 0.05
            frequency_term = 1/dt_term
        scene_data, train_traj_data, test_traj_data = load_data(scene_file, trajectory_file, dt)
        test_dataset = SE3GraspTrajDataset(scene_data, test_traj_data, pcl_root, max_history_length=8, min_future_timesteps=12, frequency=frequency_term, relative=relative, pregrasp=True, loadpcl=True, eval=True)
        concat_test_data.append(test_dataset)
    concat_test_dataset = torch.utils.data.ConcatDataset(concat_test_data)

    # test_dataset = SE3GraspTrajDataset(
    #     scene_data,
    #     test_traj_data,
    #     max_history_length=8,
    #     min_future_timesteps=12,
    #     frequency=hyperparams["frequency"],
    #     relative=True,
    #     pregrasp=True,
    #     loadpcl=True,
    #     eval=True,
    # )

    eval_dataloader = DataLoader(
        concat_test_dataset,
        collate_fn=test_dataset.collate,
        pin_memory=True,
        # batch_size=1,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
    )

    # Load model
    trajectron = Trajectron(hyperparams, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    trajectron.model.node_modules = checkpoint

    # Run evaluation
    evaluate_model(trajectron, eval_dataloader, hyperparams, args)


if __name__ == "__main__":
    main()
