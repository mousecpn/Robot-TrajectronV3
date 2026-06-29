import numpy as np
from prediction_utils import HuberCost


class ObjectGrasp:
    def __init__(self, id, grasp_pose):
        self.id = id
        self.grasp_pose = grasp_pose
        self._init_prediction_policy()

    def _init_prediction_policy(self):
        self.prediction_policy = []
        self.min_val_idx = 0
        self.prediction_policy.append(HuberCost(self.grasp_pose))

    def update_prediction_policy(self, eef_pose, eef_pose_after_action):
        for policy in self.prediction_policy:
            policy.update(eef_pose, eef_pose_after_action)
        values = [one_target_policy.get_value() for one_target_policy in self.prediction_policy]
        self.min_val_idx = np.argmin(values)

    def get_value(self):
        return self.prediction_policy[self.min_val_idx].get_value()

    def get_qvalue(self):
        return self.prediction_policy[self.min_val_idx].get_qvalue()

    def __repr__(self):
        return f"ObjectGrasp(id={self.id}, grasp_pose={self.grasp_pose})"