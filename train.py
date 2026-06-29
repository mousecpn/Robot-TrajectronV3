import torch
import numpy as np
from torch.utils.data import DataLoader
from model.trajectron import Trajectron
from tqdm import tqdm
import json
from argument_parser import args
import os
from dataset.se3_preprocessing import SE3GraspTrajDataset, load_data
import torch.nn as nn
import torch.optim as optim
import warnings
from pathlib import Path
import evaluation
torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings('ignore')

def main():
    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
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
    best_ade = 1000
    dim = 6
    relative = True

    if args.debug:
        scene_files = [Path("data/data_packed_train_raw")] 
        trajectory_files = [Path("data/trajectory/trajectories_pregrasp_packed.npz")]
        pcl_roots = [Path("data/scene_packed")]
    else:
        scene_files = [Path("data/data_pile_train_fix_raw_graspnet1b"),Path("data/data_packed_train_raw"), ] # Path("data/data_packed_train_raw")
        trajectory_files = [Path("data/trajectory/trajectories_pregrasp_pile2.npz"), Path("data/trajectory/trajectories_pregrasp_zflip.npz"), ] #   Path("data/trajectory/trajectories_pregrasp_zflip_mpc.npz")
        pcl_roots = [Path("data/scene_pile_graspnet1b"), Path("data/scene_packed"), ] #  Path("data/scene_packed")

    dt = 0.05
    frequency = 1/dt
    concat_train_data = []
    concat_test_data = []
    for scene_file, trajectory_file, pcl_root in zip(scene_files, trajectory_files, pcl_roots):
        if 'mpc' in str(trajectory_file):
            dt_term = 0.1
            frequency_term = 1/dt_term
        else:
            dt_term = 0.05
            frequency_term = 1/dt_term
        scene_data, train_traj_data, test_traj_data = load_data(scene_file, trajectory_file, dt)
        train_dataset = SE3GraspTrajDataset(scene_data, train_traj_data, pcl_root, max_history_length=8, min_future_timesteps=12, frequency=frequency_term, relative=relative, pregrasp=True, loadpcl=True, noise=True)
        test_dataset = SE3GraspTrajDataset(scene_data, test_traj_data, pcl_root, max_history_length=8, min_future_timesteps=12, frequency=frequency_term, relative=relative, pregrasp=True, loadpcl=True, eval=True)
        concat_train_data.append(train_dataset)
        concat_test_data.append(test_dataset)
    concat_train_dataset = torch.utils.data.ConcatDataset(concat_train_data)
    concat_test_dataset = torch.utils.data.ConcatDataset(concat_test_data)



    train_dataloader = DataLoader(concat_train_dataset,
                                    collate_fn=train_dataset.collate,
                                    pin_memory=True,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.preprocess_workers)
    
    eval_dataloader = DataLoader(concat_test_dataset,
                                    collate_fn=test_dataset.collate,
                                    pin_memory=True,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.preprocess_workers)
    
    hyperparams["frequency"] = frequency

    trajectron = Trajectron(hyperparams, args.device)
    os.makedirs("checkpoints", exist_ok=True)

    # args.checkpoint = "checkpoints/line20.pth"
    # trajectron = Trajectron(hyperparams, args.device)
    # model = torch.load(args.checkpoint)
    # trajectron.model.node_modules = model

    ### load pretrained model
    # pcl_backbone = torch.load("checkpoints/backbone/ptv3_encoder_epoch10.pth")
    # trajectron.model.node_modules['/pcl_encoder'] = pcl_backbone.to(args.device)
    
    
    
    trajectron.set_annealing_params()
    trajectron.model.train()
    
    optimizer = optim.Adam([
            {'params': trajectron.model.node_modules.parameters(), 
                "lr": hyperparams['learning_rate']},
        ]
    )
    # Set Learning Rate
    if hyperparams['learning_rate_style'] == 'const':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams['learning_rate_style'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=hyperparams['learning_decay_rate'])


    curr_iter = 0
    for epoch in range(1, args.train_epochs + 1):
        trajectron.model.to(args.device)
        pbar = tqdm(train_dataloader, ncols=80)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            trajectron.step_annealers()
            optimizer.zero_grad()
            train_loss = trajectron.train_loss(batch)
            pbar.set_description(f"Epoch {epoch},  L: {train_loss.item():.2f}")
            train_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model.parameters(), hyperparams['grad_clip'])
            optimizer.step()
            
            # Stepping forward the learning rate scheduler and annealers.
            if optimizer.param_groups[0]['lr'] > hyperparams['min_learning_rate']:
                lr_scheduler.step()

            curr_iter += 1
            
        print("learning_rate:",lr_scheduler.get_last_lr()[0])
        print("kl_weight:", trajectron.model.kl_weight.cpu().item())
        print("gammma:", trajectron.model.gamma.cpu().item())
        

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and epoch % args.eval_every == 0 and epoch > 0: # not args.debug and
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            trajectron.model.to(args.device)
            trajectron.model.eval()
            with torch.no_grad():
                # Calculate evaluation loss
                eval_loss_list = []
                print(f"Starting Evaluation @ epoch {epoch}")
                pbar = tqdm(eval_dataloader, ncols=80)
                ade = []
                fde = []
                for batch in pbar:
                    (first_history_index, x, q, y, context) = batch
                    eval_loss = trajectron.eval_loss(batch)
                    
                    pbar.set_description(f"Epoch {epoch}, L: {eval_loss.item():.2f}")
                    eval_loss_list.append({'nll': [eval_loss]})
                    # predictions = trajectron.predict(batch,
                    #                         ph,
                    #                         num_samples=20,
                    #                         z_mode=False,
                    #                         gmm_mode=False,
                    #                         full_dist=False)
                    predictions = trajectron.predict(batch,
                                        ph,
                                        num_samples=20,
                                        z_mode=True,
                                        gmm_mode=True,
                                        all_z_sep=True,
                                        full_dist=False)
                    
                    batch_ade = evaluation.compute_ade(predictions, y[...,0:dim]).detach().numpy()
                    batch_fde = evaluation.compute_fde(predictions, y[...,0:dim]).detach().numpy()
                    # ax = plt.axes()
                    # visualization.plot_trajectories2d(ax, predictions, x_t[0,:,0:2].detach().cpu().numpy() ,y_t[0,:,0:2].detach().cpu().numpy())


                    ade.append(batch_ade)
                    fde.append(batch_fde)
                ade = np.mean(np.concatenate(ade,axis=0))*1000
                fde = np.mean(np.concatenate(fde,axis=0))*1000
            if ade < best_ade: 
                best_ade = ade
                # model_registrar.save_models(epoch)
            model_save = trajectron.model
            torch.save(model_save.node_modules, "checkpoints/epoch{}|{}Hz|ade{:.2f}.pth".format(epoch, 20 ,ade))
            print("ade:", ade)
            print("fde:", fde)
            # navigation_evaluate(trajectron,100)

            trajectron.model.train()
            
    return


if __name__=="__main__":
    main()