# Robot-TrajectronV3

Robot-TrajectronV3 is a multimodal robot trajectory prediction pipeline for SE(3) end-effector motion. The current codebase trains a CVAE-style predictor conditioned on motion history, robot joint states, scene point clouds, and grasp candidates, then reports ADE/FDE metrics and renders 3D rollouts.

## Highlights

- SE(3) trajectory prediction with a multimodal generative model implemented in [model/mgcvae.py](/home/u0161364/clean_repo/Robot-TrajectronV3/model/mgcvae.py).
- Joint conditioning on grasp proposals and scene point clouds through the preprocessing pipeline in [dataset/se3_preprocessing.py](/home/u0161364/clean_repo/Robot-TrajectronV3/dataset/se3_preprocessing.py).
- Point cloud backbone support through Point Transformer V3 style components in [model/ptv3.py](/home/u0161364/clean_repo/Robot-TrajectronV3/model/ptv3.py).
- Built-in training, ADE/FDE evaluation, and MP4 visualization scripts via [train.py](/home/u0161364/clean_repo/Robot-TrajectronV3/train.py), [evaluate.py](/home/u0161364/clean_repo/Robot-TrajectronV3/evaluate.py), and [visualization.py](/home/u0161364/clean_repo/Robot-TrajectronV3/visualization.py).

## Installation

### Base environment (model training + evaluation + visualization)

The repository depends on a CUDA 11.8 + PyTorch 2.0.1 stack. The environment spec is captured in [environment.yml](/home/u0161364/clean_repo/Robot-TrajectronV3/environment.yml).

```bash
conda env create -f environment.yml
conda activate robot-trajectron
```

**Key runtime dependencies:**

- Core ML: `torch`, `torchvision`, `tqdm`, `tensorboard`, `tensorboardx`
- Geometry & optimization: `theseus-ai`, `scipy`
- Point cloud & 3D: `open3d`, `plyfile`, `spconv-cu118`, `torch-scatter`, `pytorch-sparse`, `pytorch-cluster`
- Data processing: `numpy`, `pandas`, `scikit-learn`, `imageio`, `pyyaml`, `h5py`
- Model utilities: `timm`, `addict`, `einops`, `sharedarray`

```bash
# Verify
python -c "import torch, open3d, imageio, theseus, torch_scatter, spconv.pytorch; print('cuda:', torch.cuda.is_available())"
```

> **Note:** `spconv-cu118` is pinned to CUDA 11.8. If your local CUDA toolchain differs, adjust this package first. The training / visualization code paths assume a CUDA-capable GPU.

### Simulation environment (Franka PyBullet deployment + benchmark)

The simulation pipeline under [simulation_experiment](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment) shares the base environment above and adds simulation-specific packages:

```bash
pip install -r simulation_experiment/requirements.txt
```

Some planning / interactive scripts additionally need:

```bash
pip install spatialmath-python spatialgeometry swift-sim pillow transformations
```

**ROS2 (optional):** Only required for the ROS2-based nodes ([trajectron_node.py](simulation_experiment/trajectron_node.py) and variants). Install a working ROS2 distribution providing `rclpy`, `geometry_msgs`, and `std_msgs`.

All simulation commands should be run from the `simulation_experiment/` directory:

```bash
cd simulation_experiment
python simulated_shared_benchmark.py --method teleop --user singledof --no-debug
```

The scripts currently expect the following repository-local layout:

```text
data/
	data_scene_raw/
	scene_data/
        point_clouds/
	trajectory/
		trajectories_pregrasp.npz
```

Training uses the packed dataset when `--debug` is enabled, and uses both packed and pile datasets otherwise.

## Data Collection

### 1. Collect trajectory data

The parallel trajectory collection script is [simulation_experiment/data_collection/collect_data_parallel.sh](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/data_collection/collect_data_parallel.sh).

```bash
cd simulation_experiment
bash data_collection/collect_data_parallel.sh
```

**Defaults:** `NUM_JOBS=10`, `SCENES_PER_JOB=2000`, entrypoint [`data_collection/main.py`](simulation_experiment/data_collection/main.py)
**Output:** `data/trajectory/trajectories_pregrasp_pile2_job_<job_id>.npz`

Single-process run:

```bash
cd simulation_experiment
python data_collection/main.py \
	--save_file_name ./data/trajectory/trajectories_pregrasp_pile2_debug.npz \
	--save_interval 1000 \
	--start-scene 0 \
	--num-scenes 100 \
	--no-debug
```

### 2. Collect point-cloud data

The parallel point-cloud collection script is [simulation_experiment/data_collection/collect_pcl_parallel.sh](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/data_collection/collect_pcl_parallel.sh).

```bash
cd simulation_experiment
bash data_collection/collect_pcl_parallel.sh
```

**Defaults:** `NUM_JOBS=18`, `SCENES_PER_JOB=2000`, entrypoint [`data_collection/pcl_collection.py`](simulation_experiment/data_collection/pcl_collection.py)
**Output:** `data/scene_data/point_clouds/`

Single-process run:

```bash
cd simulation_experiment
python data_collection/pcl_collection.py \
	--start-scene 0 \
	--num-scenes 100 \
	--no-debug
```

## Training

The main training entrypoint is [train.py](/home/u0161364/clean_repo/Robot-TrajectronV3/train.py). The checked-in example command from [train.sh](/home/u0161364/clean_repo/Robot-TrajectronV3/train.sh) is:

```bash
python train.py \
	--eval_every 1 \
	--preprocess_workers 8 \
	--batch_size 128 \
	--train_epochs 10 \
	--conf config/config.json
```

Useful flags:

- `--debug`: train only on the packed dataset branch used for quick iteration.
- `--device cuda:0`: select the GPU device.
- `--batch_size`: overrides the JSON config batch size.
- `--train_epochs`: number of epochs.

Checkpoints are saved under `checkpoints/` with names like `epoch10|20Hz|ade21.05.pth`.

## Evaluation

The offline evaluation entrypoint is [evaluate.py](/home/u0161364/clean_repo/Robot-TrajectronV3/evaluate.py).

```bash
python evaluate.py \
	--conf config/config_test.json \
	--checkpoint checkpoints/epoch10\|20Hz\|ade21.05.pth \
	--batch_size 128 \
	--device cuda:0
```

The metric implementations live in [evaluation/evaluation.py](/home/u0161364/clean_repo/Robot-TrajectronV3/evaluation/evaluation.py).

## Visualization

The rollout visualization entrypoint is [visualization.py](/home/u0161364/clean_repo/Robot-TrajectronV3/visualization.py).

```bash
python visualization.py \
	--conf config/config_test.json \
	--checkpoint checkpoints/epoch10\|20Hz\|ade21.05.pth \
	--batch_size 1 \
	--device cuda:0
```

What it does:

- Loads the packed dataset branch from `data/data_packed_train_raw`
- Reconstructs grasp candidates and scene point clouds
- Generates predicted future SE(3) trajectories
- Saves per-step frames under `gif_images/`
- Exports an MP4 per trajectory, such as `traj0.mp4`

## Configuration

The main hyperparameter files are:

- [config/config.json](/home/u0161364/clean_repo/Robot-TrajectronV3/config/config.json) for training
- [config/config_test.json](/home/u0161364/clean_repo/Robot-TrajectronV3/config/config_test.json) for evaluation and visualization

Important settings include prediction horizon, latent configuration, KL annealing, grasp and point-cloud encoders, and optimizer schedule.

## Simulation in Franka

### 1. Keyboard teleoperation and assisted control

Interactive control entrypoints:

- Direct keyboard teleoperation: [simulation_experiment/keyboard_control_scripts/teleoperation.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/keyboard_control_scripts/teleoperation.py)
- Hindsight-assisted control: [simulation_experiment/keyboard_control_scripts/ho_shared_control.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/keyboard_control_scripts/ho_shared_control.py)
- Robot-TrajectronV3 assisted control with ROS2 visualization: [simulation_experiment/keyboard_control_scripts/rtv3_shared_control.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/keyboard_control_scripts/rtv3_shared_control.py)

Typical commands:

```bash
cd simulation_experiment
python keyboard_control_scripts/teleoperation.py
python keyboard_control_scripts/ho_shared_control.py
python keyboard_control_scripts/rtv3_shared_control.py
```

The teleoperation mapping is implemented in [simulation_experiment/keyboard_control_scripts/teleoperation.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/keyboard_control_scripts/teleoperation.py):

- Arrow keys: planar translation
- `z` / `x`: vertical translation
- `j` `l` `i` `k` `u` `o`: rotational commands
- `h` / `m`: gripper actions

If you do not need ROS2, use the benchmark-oriented non-ROS2 trajectory predictor path in [simulation_experiment/trajectron_node_noros2.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/trajectron_node_noros2.py) through [simulation_experiment/simulated_shared_benchmark.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/simulated_shared_benchmark.py).

### 2. Planning scripts and planning benchmark

Planning-related entrypoints are located in [simulation_experiment/planning_scripts](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/planning_scripts).

Available scripts:

- [simulation_experiment/planning_scripts/planning_benchmark.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/planning_scripts/planning_benchmark.py)
- [simulation_experiment/planning_scripts/planning_rtv3.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/planning_scripts/planning_rtv3.py)
- [simulation_experiment/planning_scripts/planning_rtv3_parallel.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/planning_scripts/planning_rtv3_parallel.py)

Typical usage:

```bash
cd simulation_experiment
python planning_scripts/planning_benchmark.py
python planning_scripts/planning_rtv3.py
python planning_scripts/planning_rtv3_parallel.py
```

These scripts currently use repository-local defaults such as `data/data_packed_train_raw`, `pile/train`, and the checkpoints referenced inside the scripts themselves.

### 3. Simulated shared-control benchmark

The batch launcher is [simulation_experiment/simulated_shared_benchmark.sh](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/simulated_shared_benchmark.sh).

```bash
cd simulation_experiment
bash simulated_shared_benchmark.sh
```

The launcher currently sweeps:

- users: `noisy`, `laggy`, `modeswitching`, `singledof`
- method: `rt`
- `ood_alpha=0.85`
- `history_size=14`
- `sigma_coff=1`

For a single run, call [simulation_experiment/simulated_shared_benchmark.py](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/simulated_shared_benchmark.py) directly:

```bash
cd simulation_experiment
python simulated_shared_benchmark.py \
	--method rt \
	--user singledof \
	--ood_alpha 0.85 \
	--history_size 14 \
	--sigma_coff 1 \
	--no-debug
```

Supported methods:

- `teleop`: direct teleoperation baseline
- `ho`: hindsight goal-assistance baseline
- `rt`: Robot-TrajectronV3 shared control

Supported user models:

- `noisy`
- `laggy`
- `lowdof`
- `singledof`
- `modeswitching`

Default outputs are written under `./simulated_shared_benchmark_logs`.


