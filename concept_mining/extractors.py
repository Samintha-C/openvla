import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import re
import torch
from PIL import Image


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
    """
    Pass B: Geometric Concepts using GroundingDINO.
    Extracts concepts about object visibility and gripper alignment.
    """
    
    def __init__(
        self,
        model_config_path: Optional[str] = None,
        model_checkpoint_path: Optional[str] = None,
        text_threshold: float = 0.25,
        box_threshold: float = 0.3,
        alignment_pixel_threshold: float = 50.0,
        device: str = "cuda",
    ):
        """
        Initialize GroundingDINO-based geometric extractor.
        
        Args:
            model_config_path: Path to GroundingDINO config file
            model_checkpoint_path: Path to GroundingDINO checkpoint
            text_threshold: Threshold for text matching confidence
            box_threshold: Threshold for bounding box confidence
            alignment_pixel_threshold: Pixel distance threshold for gripper-target alignment
            device: Device to run model on
        """
        self.text_threshold = text_threshold
        self.box_threshold = box_threshold
        self.alignment_pixel_threshold = alignment_pixel_threshold
        self.device = device
        self.model = None
        self.tokenizer = None
        self._groundingdino_module_prefix = None
        
        if model_config_path and model_checkpoint_path:
            self._load_model(model_config_path, model_checkpoint_path)
    
    def _load_model(self, config_path: str, checkpoint_path: str):
        """Load GroundingDINO model."""
        try:
            # Try importing with lowercase first (when installed via pip)
            try:
                from groundingdino.models import build_model
                from groundingdino.util.slconfig import SLConfig
                from groundingdino.util.utils import clean_state_dict
                self._groundingdino_module_prefix = "groundingdino"
            except ImportError:
                # Fallback: try adding GroundingDINO to path if it exists but isn't installed
                import sys
                from pathlib import Path
                possible_paths = [
                    Path("/sc-cbint-vol/GroundingDINO"),
                    Path(config_path).parent.parent.parent if config_path else None,
                ]
                for gd_path in possible_paths:
                    if gd_path and gd_path.exists() and str(gd_path) not in sys.path:
                        sys.path.insert(0, str(gd_path))
                        break
                
                from GroundingDINO.groundingdino.models import build_model
                from GroundingDINO.groundingdino.util.slconfig import SLConfig
                from GroundingDINO.groundingdino.util.utils import clean_state_dict
                self._groundingdino_module_prefix = "GroundingDINO.groundingdino"
            
            args = SLConfig.fromfile(config_path)
            args.device = self.device
            self.model = build_model(args)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            self.model.eval()
            self.model.to(self.device)
            self.tokenizer = self.model.tokenizer
            print(f"Loaded GroundingDINO model on {self.device}")
        except ImportError:
            print("Warning: GroundingDINO not installed. Install with: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
            self.model = None
        except Exception as e:
            print(f"Error loading GroundingDINO model: {e}")
            self.model = None
    
    def extract_noun_phrases(self, instruction: str) -> List[str]:
        """
        Extract noun phrases from instruction that might be target objects.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            List of potential target object phrases
        """
        instruction_lower = instruction.lower()
        
        patterns = [
            r"pick up (?:the )?([a-z]+(?: [a-z]+)*)",
            r"grab (?:the )?([a-z]+(?: [a-z]+)*)",
            r"move (?:the )?([a-z]+(?: [a-z]+)*)",
            r"place (?:the )?([a-z]+(?: [a-z]+)*)",
            r"put (?:the )?([a-z]+(?: [a-z]+)*)",
            r"([a-z]+(?: [a-z]+)*) (?:on|onto|in|into)",
        ]
        
        objects = []
        for pattern in patterns:
            matches = re.findall(pattern, instruction_lower)
            objects.extend(matches)
        
        if not objects:
            words = instruction_lower.split()
            if len(words) > 2:
                objects.append(" ".join(words[-2:]))
            elif len(words) > 0:
                objects.append(words[-1])
        
        return list(set(objects))
    
    def detect_objects(
        self,
        image: Image.Image,
        prompt: str,
    ) -> Tuple[List[Dict], Optional[torch.Tensor]]:
        """
        Detect objects in image using GroundingDINO.
        
        Args:
            image: PIL Image
            prompt: Text prompt describing objects to detect
            
        Returns:
            Tuple of (detections list, image tensor)
        """
        if self.model is None:
            return [], None
        
        try:
            # Use the same import path that was used during model loading
            if self._groundingdino_module_prefix == "groundingdino":
                from groundingdino.datasets.transforms import Compose, Normalize, Resize, ToTensor
            elif self._groundingdino_module_prefix == "GroundingDINO.groundingdino":
                from GroundingDINO.groundingdino.datasets.transforms import Compose, Normalize, Resize, ToTensor
            else:
                # Fallback: try both
                try:
                    from groundingdino.datasets.transforms import Compose, Normalize, Resize, ToTensor
                except ImportError:
                    from GroundingDINO.groundingdino.datasets.transforms import Compose, Normalize, Resize, ToTensor
            
            transform = Compose([
                Resize([800], max_size=1333),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensor(),
            ])
            
            image_tensor, _ = transform(image, None)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor, captions=[prompt])
            
            logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"][0]  # (nq, 4)
            
            detections = []
            for i in range(logits.shape[0]):
                max_logit = logits[i].max().item()
                if max_logit > self.box_threshold:
                    box = boxes[i].cpu().numpy()
                    detections.append({
                        "box": box,
                        "confidence": max_logit,
                        "logits": logits[i].cpu().numpy(),
                    })
            
            return detections, image_tensor
        except Exception as e:
            print(f"Error in object detection: {e}")
            return [], None
    
    def compute_gripper_position(
        self,
        proprio: np.ndarray,
        image_shape: Tuple[int, int],
        camera_params: Optional[Dict] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Estimate gripper position in image coordinates from proprioceptive state.
        
        Args:
            proprio: Proprioceptive state vector
            image_shape: (height, width) of image
            camera_params: Optional camera calibration parameters
            
        Returns:
            (x, y) pixel coordinates of gripper, or None if cannot estimate
        """
        if len(proprio) < 3:
            return None
        
        eef_pos_3d = proprio[:3]
        
        if camera_params:
            K = camera_params.get("intrinsic_matrix")
            if K is not None:
                eef_pos_homogeneous = np.array([eef_pos_3d[0], eef_pos_3d[1], eef_pos_3d[2], 1.0])
                pixel_pos = K @ eef_pos_homogeneous[:3]
                pixel_pos = pixel_pos / pixel_pos[2]
                return (float(pixel_pos[0]), float(pixel_pos[1]))
        
        return None
    
    def extract(
        self,
        image: Image.Image,
        instruction: str,
        proprio: Optional[np.ndarray] = None,
        camera_params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Extract geometric concepts from image and instruction.
        
        Args:
            image: PIL Image
            instruction: Natural language instruction
            proprio: Optional proprioceptive state for gripper position
            camera_params: Optional camera calibration parameters
            
        Returns:
            Dictionary of concept values
        """
        concepts = {
            "target_visible": 0.0,
            "aligned_with_target": 0.0,
        }
        
        if self.model is None:
            return concepts
        
        noun_phrases = self.extract_noun_phrases(instruction)
        if not noun_phrases:
            return concepts
        
        prompt = " . ".join(noun_phrases) + " . gripper"
        
        detections, _ = self.detect_objects(image, prompt)
        
        if not detections:
            return concepts
        
        target_detections = [d for d in detections if d["confidence"] > self.box_threshold]
        if target_detections:
            best_detection = max(target_detections, key=lambda x: x["confidence"])
            concepts["target_visible"] = 1.0
            
            if proprio is not None:
                box = best_detection["box"]
                box_center_x = (box[0] + box[2]) / 2 * image.width
                box_center_y = (box[1] + box[3]) / 2 * image.height
                
                gripper_pos = self.compute_gripper_position(proprio, image.size, camera_params)
                if gripper_pos:
                    distance = np.sqrt(
                        (gripper_pos[0] - box_center_x) ** 2 +
                        (gripper_pos[1] - box_center_y) ** 2
                    )
                    if distance < self.alignment_pixel_threshold:
                        concepts["aligned_with_target"] = 1.0
        
        return concepts

