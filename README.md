# Robot-TrajectronV3

**Robot-TrajectronV3** is a multimodal robot trajectory prediction pipeline for SE(3) end-effector motion. It trains a CVAE-style predictor conditioned on motion history, robot joint states, scene point clouds, and grasp candidates — then reports ADE/FDE metrics and renders slick 3D rollouts. All on a Franka Emika Panda arm! 

[[Webpage]](https://mousecpn.github.io/RTV3_page/)

## ✨ Highlights

- **SE(3) shared control via Bayesian posterior inference** — real-time intent inference from user input combined with a learned trajectory prior, producing assistive actions on the Franka Panda (see [`simulation_experiment/simulated_shared_benchmark.py`](simulation_experiment/simulated_shared_benchmark.py) and [`trajectron_node_noros2.py`](simulation_experiment/trajectron_node_noros2.py)).
- **Trajectory dataset generation** — automated Franka PyBullet pipeline for collecting large-scale SE(3) end-effector trajectories with joint states, grasp candidates, and scene point clouds (entrypoints in [`simulation_experiment/data_collection/`](simulation_experiment/data_collection/)).
- **Goal-reaching simulation benchmark** — standardized evaluation of shared-control policies (teleop, hindsight, RTV3) under diverse simulated user types (noisy, laggy, single-DoF, mode-switching) with success rate and collision metrics.

## 🔧 Installation

### 📦 Base environment (model training + evaluation + visualization)

The repository depends on a CUDA 11.8 + PyTorch 2.0.1 stack. The environment spec is captured in [environment.yml](/home/u0161364/clean_repo/Robot-TrajectronV3/environment.yml).

```bash
conda env create -f environment.yml
conda activate robot-trajectron
MAX_JOBS=4 pip install flash-attn==2.3.6 --no-build-isolation
```

**📋 Key runtime dependencies:**

- Core ML: `torch`, `torchvision`, `tqdm`, `tensorboard`, `tensorboardx`
- Geometry & optimization: `theseus-ai`, `scipy`
- Point cloud & 3D: `open3d`, `plyfile`, `spconv-cu118`, `torch-scatter`, `pytorch-sparse`, `pytorch-cluster`
- Point Transformer V3: `spconv-cu118`, `torch-scatter`, `flash-attn`, `timm`, `addict`
- Data processing: `numpy`, `pandas`, `scikit-learn`, `imageio`, `pyyaml`, `h5py`
- Model utilities: `timm`, `addict`, `einops`, `sharedarray`

```bash
# ✅ Verify installation
python -c "import torch, open3d, imageio, theseus, torch_scatter, spconv.pytorch; print('cuda:', torch.cuda.is_available())"
```

> **💡 Note:** `spconv-cu118` is pinned to CUDA 11.8. If your local CUDA toolchain differs, adjust this package first. The training / visualization code paths assume a CUDA-capable GPU.

> **💡 Note (PTv3 / FlashAttention):** The Point Transformer V3 backbone ([model/ptv3.py](model/ptv3.py)) uses FlashAttention for efficient training. `flash-attn` is listed in `environment.yml` and will be compiled during `conda env create`. This requires **CUDA ≥ 11.6** and a compatible GPU. If your environment doesn't satisfy this, the model falls back gracefully (`import flash_attn` is wrapped in `try/except`).

### 🎮 Simulation environment (Franka PyBullet deployment + benchmark)

The simulation pipeline under [simulation_experiment](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment) shares the base environment above and adds simulation-specific packages:

```bash
pip install -r simulation_experiment/requirements.txt
```

Some planning / interactive scripts additionally need:

```bash
pip install spatialmath-python spatialgeometry swift-sim pillow transformations
```

**� CuRobo (motion planner):** The default motion planner in [`ClutterRemovalSim`](simulation_experiment/experiment/simulation.py) is [**CuRobo**](https://github.com/nvlabs/curobo) from NVIDIA Labs. Install it with:

```bash
pip install curobo
```

> **💡 Note:** CuRobo requires **CUDA ≥ 11.8** and a compatible GPU. If unavailable, you can switch to the `neo_ss` or `mppi` fallback planners by passing `planning='neo_ss'` (see [`experiment/simulation.py`](simulation_experiment/experiment/simulation.py) for details).

**�🟢 ROS2 (optional):** Only required for the ROS2-based nodes ([trajectron_node.py](simulation_experiment/trajectron_node.py) and variants). Install a working ROS2 distribution providing `rclpy`, `geometry_msgs`, and `std_msgs`.

All simulation commands should be run from the `simulation_experiment/` directory:

```bash
cd simulation_experiment
# 🚀 Quick smoke test
python simulated_shared_benchmark.py --method teleop --user singledof --no-debug
```

The scripts currently expect the following repository-local layout:

```text
data/
	urdfs/                   ← robot URDF files (Franka Panda)
	data_scene_raw/          ← grasp scene data
	scene_data/
        point_clouds/
	trajectory/
		trajectories_pregrasp.npz
```

> **📥 Download required assets:** Some directories above are **not** included in the repository and must be downloaded separately:
> - **`data/urdfs/`** — Franka Panda robot URDF & meshes: [Download from Google Drive](https://drive.google.com/file/d/12o0RlOqypwNL8a3RSuSnlf3c0zhig6Fi/view?usp=drive_link) → extract into `data/urdfs/`
> - **`data/data_scene_raw/`** — Grasp scene data (grasps.csv, mesh_pose_list, scenes): [Download from Google Drive](https://drive.google.com/file/d/1UWJUufqldwXkl1FPecM07PyVpGk4fws9/view?usp=sharing) → extract into `data/data_scene_raw/`
>
> After downloading, verify the contents:
> ```bash
> ls data/urdfs/           # should contain .urdf and .obj/.stl files
> ls data/data_scene_raw/  # should contain grasps.csv, mesh_pose_list/, scenes/
> ```

Training uses the packed dataset when `--debug` is enabled, and uses both packed and pile datasets otherwise.

## 📊 Data Collection

### 🏃 1. Collect trajectory data

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

### ☁️ 2. Collect point-cloud data

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

## 🏋️ Training

The main training entrypoint is [train.py](/home/u0161364/clean_repo/Robot-TrajectronV3/train.py). The checked-in example command from [train.sh](/home/u0161364/clean_repo/Robot-TrajectronV3/train.sh) is:

```bash
python train.py \
	--eval_every 1 \
	--preprocess_workers 8 \
	--batch_size 128 \
	--train_epochs 10 \
	--conf config/config.json
```

🎛️ Useful flags:

- `--debug`: train only on the packed dataset branch used for quick iteration.
- `--device cuda:0`: select the GPU device.
- `--batch_size`: overrides the JSON config batch size.
- `--train_epochs`: number of epochs.
- `--scene_files`, `--trajectory_files`, `--pcl_roots`: override default data paths (see below).

**📂 Custom data paths:** All scripts accept `--scene_files`, `--trajectory_files`, and `--pcl_roots` as space-separated lists. Defaults match the repository layout:

```bash
# Full dataset (default, non-debug mode)
python train.py ... \
	--scene_files data/data_pile_train_fix_raw_graspnet1b data/data_scene_raw \
	--trajectory_files data/trajectory/trajectories_pregrasp_pile2.npz data/trajectory/trajectories_pregrasp_zflip.npz \
	--pcl_roots data/scene_pile_graspnet1b data/scene_data

# Debug / packed-only
python train.py ... --debug
# (--debug automatically uses packed-only paths; you can also pass them explicitly)
```

Checkpoints are saved under `checkpoints/` with names like `epoch10|20Hz|ade21.05.pth`.

## 📏 Evaluation

The offline evaluation entrypoint is [evaluate.py](/home/u0161364/clean_repo/Robot-TrajectronV3/evaluate.py).

```bash
python evaluate.py \
	--conf config/config_test.json \
	--checkpoint checkpoints/epoch10\|20Hz\|ade21.05.pth \
	--batch_size 128 \
	--device cuda:0
```

The metric implementations live in [evaluation/evaluation.py](/home/u0161364/clean_repo/Robot-TrajectronV3/evaluation/evaluation.py).

## 🎬 Visualization

The rollout visualization entrypoint is [visualization.py](/home/u0161364/clean_repo/Robot-TrajectronV3/visualization.py).

```bash
python visualization.py \
	--conf config/config_test.json \
	--checkpoint checkpoints/epoch10\|20Hz\|ade21.05.pth \
	--batch_size 1 \
	--device cuda:0
```

🎯 What it does:

- Loads scene + trajectory + point-cloud data (controlled by `--scene_files`, `--trajectory_files`, `--pcl_roots`; takes the first entry of each)
- Defaults to the packed dataset branch (`data/data_scene_raw` / `trajectories_pregrasp.npz` / `data/scene_data`)
- Reconstructs grasp candidates and scene point clouds
- Generates predicted future SE(3) trajectories
- Saves per-step frames under `gif_images/`
- Exports an MP4 per trajectory, such as `traj0.mp4`

## ⚙️ Configuration

The main hyperparameter files are:

- [config/config.json](/home/u0161364/clean_repo/Robot-TrajectronV3/config/config.json) for training
- [config/config_test.json](/home/u0161364/clean_repo/Robot-TrajectronV3/config/config_test.json) for evaluation and visualization

Important settings include prediction horizon, latent configuration, KL annealing, grasp and point-cloud encoders, and optimizer schedule.

## 🤖 Simulation in Franka

### ⌨️ 1. Keyboard teleoperation and assisted control

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

### 🗺️ 2. Planning scripts and planning benchmark

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

These scripts currently use repository-local defaults such as `data/data_scene_raw`, `pile/train`, and the checkpoints referenced inside the scripts themselves.

### 🏆 3. Simulated shared-control benchmark

The batch launcher is [simulation_experiment/simulated_shared_benchmark.sh](/home/u0161364/clean_repo/Robot-TrajectronV3/simulation_experiment/simulated_shared_benchmark.sh).

```bash
cd simulation_experiment
bash simulated_shared_benchmark.sh
```



The launcher currently sweeps:

- users: `noisy`, `laggy`, `modeswitching`, `singledof`
- method: `rt`, `ho`, `teleop`


https://github.com/user-attachments/assets/129f2c96-6a72-4b0a-a748-1e92cea9ef8b



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

🎮 Supported methods:

- `teleop` — direct teleoperation baseline
- `ho` — hindsight goal-assistance baseline
- `rt` — Robot-TrajectronV3 shared control

👤 Supported user models:

- `noisy` 
- `laggy` 
- `lowdof` 
- `singledof` 
- `modeswitching` 

Default outputs are written under `./simulated_shared_benchmark_logs`.


