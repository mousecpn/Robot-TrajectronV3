# Quick Start Guide - Updated Scripts

## Main Simulation Script

### simulated_shared_benchmark.py

Run the simulated shared control benchmark with various methods and user types.

#### Basic Usage

```bash
# Teleoperation with single DOF user
python simulated_shared_benchmark.py --method teleop --user singledof

# Robot Trajectron method with noisy user
python simulated_shared_benchmark.py --method rt --user noisy

# Hindsight method with laggy user
python simulated_shared_benchmark.py --method ho --user laggy
```

#### Custom Configuration

```bash
# Using custom data and checkpoint paths
python simulated_shared_benchmark.py \
    --method rt \
    --user singledof \
    --data_root data/custom_scene_data \
    --checkpoint checkpoints/my_model.pth \
    --log_dir ./my_logs \
    --scene_type custom_pile \
    --scene_path custom_pile/train
```

#### Model Parameters

```bash
# Adjust model hyperparameters
python simulated_shared_benchmark.py \
    --method rt \
    --user modeswitching \
    --ood_alpha 0.85 \
    --history_size 8 \
    --sigma_coff 1.2
```

#### Debug Mode

```bash
# Run without GUI/debug output
python simulated_shared_benchmark.py --method teleop --user noisy --no-debug
```

### Available Options

**Methods:**
- `rt`: Robot Trajectron (trajectory prediction)
- `ho`: Hindsight (goal prediction)
- `teleop`: Direct teleoperation (baseline)

**User Types:**
- `noisy`: User with noisy inputs
- `laggy`: User with input lag
- `lowdof`: User controlling fewer DOFs
- `singledof`: User controlling one DOF at a time
- `modeswitching`: User switching between control modes

**Model Parameters:**
- `--ood_alpha`: Out-of-distribution detection threshold (0-1, default: 0.9)
- `--history_size`: Number of past states to consider (default: 6)
- `--sigma_coff`: Uncertainty coefficient (default: 1.0)

### Example Experiments

**Compare methods with same user:**
```bash
python simulated_shared_benchmark.py --method teleop --user singledof
python simulated_shared_benchmark.py --method rt --user singledof
python simulated_shared_benchmark.py --method ho --user singledof
```

**Compare users with same method:**
```bash
python simulated_shared_benchmark.py --method rt --user noisy
python simulated_shared_benchmark.py --method rt --user laggy
python simulated_shared_benchmark.py --method rt --user singledof
```

## Path Configuration

All paths are now relative by default. The parent directory (Robot-TrajectronV3) is automatically added to Python's path.

### Default Path Structure

```
Robot-TrajectronV3/
├── config/
│   ├── config.json
│   └── config_test.json
├── checkpoints/
│   ├── line24.pth
│   └── ...
├── simulation_experiment/  (current directory)
│   ├── data/
│   │   └── ycb_scene_packed/
│   ├── simulated_shared_benchmark.py
│   └── ...
└── ...
```

### Custom Paths

If your structure is different, use the path arguments:

```bash
python simulated_shared_benchmark.py \
    --data_root /absolute/path/to/data \
    --checkpoint /absolute/path/to/model.pth \
    --log_dir /absolute/path/to/logs
```

## External Dependencies

### sdfsc (Collision Checker)

The collision checker is an external dependency. Set it up:

```bash
# Option 1: Default location
git clone <sdfsc-repo> ~/sdfsc

# Option 2: Custom location
export SDFSC_PATH=/path/to/sdfsc
python simulated_shared_benchmark.py ...
```

If not installed, the scripts will run but collision checking may not work properly.

## Logging

Logs are saved to the directory specified by `--log_dir` (default: `./simulated_shared_benchmark_logs`).

Results include:
- Success rate
- Collision rate  
- Average iterations per grasp
- Total count of attempts

## Tips

1. **Start with teleop**: Test basic functionality first
   ```bash
   python simulated_shared_benchmark.py --method teleop --user singledof --no-debug
   ```

2. **Use small datasets**: For testing, use a smaller data_root
   
3. **Monitor GPU memory**: The scripts use CUDA - ensure your GPU has sufficient memory

4. **Check paths**: If imports fail, verify the directory structure matches expectations

5. **Debug mode**: Use debug mode (without --no-debug) to see more detailed output

## Troubleshooting

**Import errors:**
- Ensure you're running from the `simulation_experiment` directory
- Check that parent directory contains the required modules

**CUDA errors:**
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory: `nvidia-smi`

**Path not found:**
- Use absolute paths if relative paths don't work
- Verify data exists at specified locations

**Missing dependencies:**
- Install requirements: `pip install -r requirements.txt`
- Set SDFSC_PATH for collision checker
