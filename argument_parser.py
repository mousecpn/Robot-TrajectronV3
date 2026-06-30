import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    type=str,
                    default='config/config.json')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=0)


# Data Parameters
parser.add_argument("--scene_files",
                    help="space-separated list of scene data directories",
                    type=str, nargs='+',
                    default=["data/data_pile_train_fix_raw_graspnet1b", "data/data_packed_train_raw"])

parser.add_argument("--trajectory_files",
                    help="space-separated list of trajectory .npz files",
                    type=str, nargs='+',
                    default=["data/trajectory/trajectories_pregrasp_pile2.npz", "data/trajectory/trajectories_pregrasp_zflip.npz"])

parser.add_argument("--pcl_roots",
                    help="space-separated list of point cloud root directories",
                    type=str, nargs='+',
                    default=["data/scene_pile_graspnet1b", "data/scene_packed"])

parser.add_argument("--data_path",
                    help="json",
                    type=str,
                    default="/home/data/data_from_root/data/data_bmi2d_w_goals_colavoid_1m.json")

parser.add_argument("--checkpoint",
                    help="the checkpoint file",
                    type=str,
                    default="checkpoints/Exp42_maxent_autoalpha_93.pth")


parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')

# Training Parameters
parser.add_argument("--train_epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=1)

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=128)


parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)


args = parser.parse_args()
