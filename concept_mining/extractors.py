import numpy as np
from typing import Dict, Any, Optional


class ProprioceptiveExtractor:
    def __init__(
        self,
        gripper_closed_threshold: float = 0.05,
        arm_moving_epsilon: float = 0.01,
        table_height: float = 0.0,
    ):
        self.gripper_closed_threshold = gripper_closed_threshold
        self.arm_moving_epsilon = arm_moving_epsilon
        self.table_height = table_height
        self.prev_eef_pos = None

    def extract(self, proprio: np.ndarray, state_encoding: str = "POS_EULER") -> Dict[str, float]:
        """
        Extract proprioceptive concepts from robot state.
        
        Args:
            proprio: Proprioceptive state vector (8-dim for POS_EULER: XYZ(3) + RPY(3) + PAD(1) + Gripper(1))
            state_encoding: Type of state encoding (POS_EULER, POS_QUAT, JOINT)
        
        Returns:
            Dictionary of concept values
        """
        concepts = {}
        
        if state_encoding == "POS_EULER":
            eef_pos = proprio[:3]
            gripper_state = proprio[-1]
            
            concepts["gripper_closed"] = 1.0 if gripper_state < self.gripper_closed_threshold else 0.0
            
            if self.prev_eef_pos is not None:
                delta_pos = np.linalg.norm(eef_pos - self.prev_eef_pos)
                concepts["arm_moving"] = 1.0 if delta_pos > self.arm_moving_epsilon else 0.0
            else:
                concepts["arm_moving"] = 0.0
            
            concepts["height_above_table"] = 1.0 if eef_pos[2] > self.table_height else 0.0
            
            self.prev_eef_pos = eef_pos.copy()
            
        elif state_encoding == "POS_QUAT":
            eef_pos = proprio[:3]
            gripper_state = proprio[-1]
            
            concepts["gripper_closed"] = 1.0 if gripper_state < self.gripper_closed_threshold else 0.0
            
            if self.prev_eef_pos is not None:
                delta_pos = np.linalg.norm(eef_pos - self.prev_eef_pos)
                concepts["arm_moving"] = 1.0 if delta_pos > self.arm_moving_epsilon else 0.0
            else:
                concepts["arm_moving"] = 0.0
            
            concepts["height_above_table"] = 1.0 if eef_pos[2] > self.table_height else 0.0
            
            self.prev_eef_pos = eef_pos.copy()
            
        elif state_encoding == "JOINT":
            gripper_state = proprio[-1]
            joint_pos = proprio[:-1]
            
            concepts["gripper_closed"] = 1.0 if gripper_state < self.gripper_closed_threshold else 0.0
            
            if self.prev_eef_pos is not None:
                delta_pos = np.linalg.norm(joint_pos - self.prev_eef_pos)
                concepts["arm_moving"] = 1.0 if delta_pos > self.arm_moving_epsilon else 0.0
            else:
                concepts["arm_moving"] = 0.0
            
            concepts["height_above_table"] = 0.0
            
            self.prev_eef_pos = joint_pos.copy()
        else:
            concepts["gripper_closed"] = 0.0
            concepts["arm_moving"] = 0.0
            concepts["height_above_table"] = 0.0
        
        return concepts

    def reset(self):
        self.prev_eef_pos = None

