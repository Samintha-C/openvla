import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import re
try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    Image = None


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


class GeometricExtractor:
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        alignment_pixel_threshold: int = 50,
        device: Optional[str] = None,
    ):
        """
        Extract geometric concepts using GroundingDINO.
        
        Args:
            confidence_threshold: Minimum confidence for object detection
            alignment_pixel_threshold: Maximum pixel distance for alignment
            device: Device to run GroundingDINO on
        """
        self.confidence_threshold = confidence_threshold
        self.alignment_pixel_threshold = alignment_pixel_threshold
        if device is None:
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load GroundingDINO model."""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. GroundingDINO requires PyTorch.")
            self.model = None
            return
        
        try:
            from groundingdino.util.inference import load_model, load_image, predict
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
            import groundingdino.datasets.transforms as T
            
            self.gdino_load_model = load_model
            self.gdino_load_image = load_image
            self.gdino_predict = predict
            self.gdino_SLConfig = SLConfig
            self.gdino_clean_state_dict = clean_state_dict
            self.gdino_transforms = T
            
            config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_path = "groundingdino_swint_ogc.pth"
            
            if not Path(weights_path).exists():
                print(f"Warning: GroundingDINO weights not found at {weights_path}")
                print("Please download from: https://github.com/IDEA-Research/GroundingDINO/releases")
                self.model = None
                return
            
            args = self.gdino_SLConfig.fromfile(config_file)
            args.device = self.device
            model = self.gdino_load_model(args, weights_path)
            model.eval()
            self.model = model
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            print(f"GroundingDINO loaded on {self.device}")
        except ImportError as e:
            print(f"Warning: GroundingDINO not available: {e}")
            print("Install with: pip install groundingdino-py")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load GroundingDINO: {e}")
            self.model = None
    
    def parse_instruction(self, instruction: str) -> List[str]:
        """
        Parse instruction to extract target objects.
        
        Examples:
            "pick up the red can" -> ["red can", "can"]
            "move the cup to the table" -> ["cup"]
            "grasp the bottle" -> ["bottle"]
        """
        if not instruction:
            return []
        
        instruction = instruction.lower().strip()
        
        patterns = [
            r"(?:pick up|grasp|grab|take|get|move|place|put)\s+(?:the\s+)?([a-z\s]+?)(?:\s+to|\s+on|\s+in|$)",
            r"the\s+([a-z\s]+?)(?:\s+to|\s+on|\s+in|$)",
        ]
        
        objects = []
        for pattern in patterns:
            matches = re.findall(pattern, instruction)
            for match in matches:
                obj = match.strip()
                if obj and len(obj.split()) <= 3:
                    objects.append(obj)
        
        if not objects:
            words = instruction.split()
            if len(words) >= 3:
                objects.append(" ".join(words[-2:]))
        
        return list(set(objects)) if objects else []
    
    def extract(
        self,
        image: np.ndarray,
        instruction: str,
        gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, float]:
        """
        Extract geometric concepts from image and instruction.
        
        Args:
            image: RGB image as numpy array (H, W, 3) uint8
            instruction: Language instruction
            gripper_bbox: Optional gripper bounding box (x1, y1, x2, y2) in pixels
        
        Returns:
            Dictionary of concept values:
            - target_visible: 1.0 if target object detected with confidence > threshold
            - aligned_with_target: 1.0 if gripper aligned with target (within threshold)
        """
        concepts = {
            "target_visible": 0.0,
            "aligned_with_target": 0.0,
        }
        
        if self.model is None:
            return concepts
        
        try:
            target_objects = self.parse_instruction(instruction)
            if not target_objects:
                return concepts
            
            text_prompt = ". ".join(target_objects) + "."
            
            img_pil = Image.fromarray(image)
            img_transformed, _ = self.transform(img_pil, None)
            
            with torch.no_grad():
                boxes, logits, phrases = self.gdino_predict(
                    model=self.model,
                    image=img_transformed,
                    caption=text_prompt,
                    box_threshold=self.confidence_threshold,
                    text_threshold=self.confidence_threshold,
                )
            
            if len(boxes) > 0:
                max_conf_idx = logits.argmax().item()
                target_box = boxes[max_conf_idx].cpu().numpy()
                target_confidence = logits[max_conf_idx].item()
                
                if target_confidence >= self.confidence_threshold:
                    concepts["target_visible"] = 1.0
                    
                    if gripper_bbox is not None:
                        target_center = np.array([
                            (target_box[0] + target_box[2]) / 2,
                            (target_box[1] + target_box[3]) / 2,
                        ])
                        gripper_center = np.array([
                            (gripper_bbox[0] + gripper_bbox[2]) / 2,
                            (gripper_bbox[1] + gripper_bbox[3]) / 2,
                        ])
                        
                        distance = np.linalg.norm(target_center - gripper_center)
                        if distance <= self.alignment_pixel_threshold:
                            concepts["aligned_with_target"] = 1.0
        
        except Exception as e:
            print(f"Error in geometric extraction: {e}")
        
        return concepts

