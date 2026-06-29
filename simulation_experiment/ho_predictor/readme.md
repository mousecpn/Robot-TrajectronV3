# Goal Predictor

HO prediction method.

## Files

### prediction_utils.py
Core utility functions for prediction:
- `HuberCost`: Cost function class for goal prediction
- `ApplyTwistToTransform`: Apply twist to transformation matrix
- `QuaternionDistance`: Calculate quaternion distance

### object_grasp.py
Single grasp object class:
- `ObjectGrasp`: Represents a grasp pose with prediction policy (can be modified to multiple grasps for one object)

### grasps_predcition.py
Main predictor class:
- `GraspsPredictor`: Manages multiple grasp targets and predictions

## Usage

```python
import numpy as np
from grasps_predcition import GraspsPredictor

# Create grasp poses (4x4 matrices)
grasp_poses = [np.eye(4), np.eye(4)]  # Add your poses

# Initial robot pose
eef_pose = np.eye(4)
eef_pose[0:3, 3] = [0.5, 0.0, 0.3]  # Set position

# Create predictor
predictor = GraspsPredictor(grasp_poses, eef_pose)

# Update with robot action
twist = [0.1, 0.0, 0.0, 0.0, 0.0, 0.1]  # [linear_vel, angular_vel]
predictor.update_prediction_policies(eef_pose, twist)

# Get predictions
goal_id, goal_grasp = predictor.get_goal_grasp()
predictor.print_probability()
```

## Test

```bash
python grasps_predcition.py
```

Generates random grasp poses and demonstrates prediction updates.