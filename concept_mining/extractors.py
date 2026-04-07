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
    """Extract geometric concepts using GroundingDINO + proprioception.

    Produces per-frame spatial concepts designed to feed a linear concept
    bottleneck layer. Target position comes from GroundingDINO (run at
    trajectory start and on gripper-state changes); EEF position comes from
    per-frame proprioception, so spatial concepts vary every frame. An
    optional monocular depth estimate augments the target with a z-proxy.

    The extractor does NOT compute precomputed differences (e.g. dx_eef_target)
    because the camera-to-robot axis mapping is unknown. Instead it emits raw
    normalized EEF and target positions; a downstream linear layer can learn
    the mapping.
    """

    CONCEPT_KEYS: List[str] = [
        "target_visible",
        "target_cx",
        "target_cy",
        "target_bbox_size",
        "target_depth",
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_delta_x",
        "eef_delta_y",
        "eef_delta_z",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
        use_depth: bool = False,
        depth_model_name: str = "LiheYoung/depth-anything-v2-small-hf",
        workspace_x_min: float = 0.1,
        workspace_x_max: float = 0.4,
        workspace_y_min: float = -0.15,
        workspace_y_max: float = 0.25,
        workspace_z_min: float = 0.0,
        workspace_z_max: float = 0.3,
    ):
        """
        Args:
            confidence_threshold: Minimum confidence for GroundingDINO detections.
            device: Device to run GroundingDINO / depth on.
            use_depth: If True, load DepthAnythingV2 for target z-proxy.
            depth_model_name: HF model name for monocular depth estimator.
            workspace_{x,y,z}_{min,max}: Empirical EEF workspace bounds used to
                normalize proprio XYZ into [0, 1]. Defaults tuned for
                BridgeDataV2 WidowX.
        """
        self.confidence_threshold = confidence_threshold
        if device is None:
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.use_depth = use_depth
        self.depth_model_name = depth_model_name

        self.workspace_x_min = workspace_x_min
        self.workspace_x_max = workspace_x_max
        self.workspace_y_min = workspace_y_min
        self.workspace_y_max = workspace_y_max
        self.workspace_z_min = workspace_z_min
        self.workspace_z_max = workspace_z_max

        self.model = None
        self.depth_estimator = None
        self._load_model()
        if self.use_depth:
            self._load_depth_model()
        self.reset()
        self.reset_stats()

    def reset(self) -> None:
        """Clear all per-trajectory state. Call between trajectories."""
        self.target_box: Optional[np.ndarray] = None       # [cx, cy, w, h] normalized
        self.target_depth: Optional[float] = None          # normalized [0, 1]
        self.prev_proprio: Optional[np.ndarray] = None     # for delta concepts

    def reset_stats(self) -> None:
        """Reset cumulative diagnostic counters. Call once before a mining run."""
        self._target_conf_sum: float = 0.0
        self._target_conf_count: int = 0
        self._target_detection_attempts: int = 0
        self._target_detection_failures: int = 0
        self._frames_with_proprio: int = 0
        self._frames_without_proprio: int = 0

    def get_stats(self) -> dict:
        """Return cumulative diagnostic stats across all processed frames."""
        mean_conf = self._target_conf_sum / self._target_conf_count if self._target_conf_count > 0 else 0.0
        fail_pct = 100.0 * self._target_detection_failures / self._target_detection_attempts if self._target_detection_attempts > 0 else 0.0
        return {
            "target_detection_attempts": self._target_detection_attempts,
            "target_detection_failures": self._target_detection_failures,
            "target_detection_fail_pct": fail_pct,
            "target_detections": self._target_conf_count,
            "target_mean_conf": mean_conf,
            "frames_with_proprio": self._frames_with_proprio,
            "frames_without_proprio": self._frames_without_proprio,
            # Back-compat keys for existing logging
            "gripper_detected": 0,
            "gripper_fallback": 0,
            "gripper_fallback_pct": 0.0,
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load GroundingDINO model."""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. GroundingDINO requires PyTorch.")
            self.model = None
            return

        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T

            self.gdino_predict = predict

            config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_path = "groundingdino_swint_ogc.pth"

            if not Path(weights_path).exists():
                print(f"Warning: GroundingDINO weights not found at {weights_path}")
                print("Please download from: https://github.com/IDEA-Research/GroundingDINO/releases")
                self.model = None
                return

            if not Path(config_file).exists():
                print(f"Warning: GroundingDINO config file not found at {config_file}")
                self.model = None
                return

            model = load_model(config_file, weights_path)
            model = model.to(self.device)
            model.eval()
            self.model = model
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            print(f"GroundingDINO loaded on {self.device}")
        except ImportError as e:
            print(f"Warning: GroundingDINO not available: {e}")
            print("Install with: pip install groundingdino-py")
            self.model = None
        except Exception as e:
            print(f"Warning: Failed to load GroundingDINO: {e}")
            self.model = None

    # ------------------------------------------------------------------
    # Depth model loading
    # ------------------------------------------------------------------

    def _load_depth_model(self) -> None:
        """Load DepthAnythingV2 via HF pipeline. Sets self.depth_estimator or leaves None."""
        if not TORCH_AVAILABLE:
            return
        try:
            from transformers import pipeline
            self.depth_estimator = pipeline(
                "depth-estimation",
                model=self.depth_model_name,
                device=0 if self.device == "cuda" else -1,
            )
            print(f"Depth estimator loaded: {self.depth_model_name} on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load depth model ({self.depth_model_name}): {e}")
            self.depth_estimator = None

    # ------------------------------------------------------------------
    # Helper: raw GroundingDINO detection
    # ------------------------------------------------------------------

    def _detect_objects(
        self, image: np.ndarray, text_prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run GroundingDINO detection.

        Returns:
            (boxes, confidences, phrases) where boxes is (N, 4) normalised
            cxcywh. Empty arrays if nothing detected.
        """
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

        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), []

        return boxes.cpu().numpy(), logits.cpu().numpy(), phrases

    # ------------------------------------------------------------------
    # Target detection (called at trajectory start + gripper state changes)
    # ------------------------------------------------------------------

    def detect_target(self, image: np.ndarray, instruction: str) -> Optional[np.ndarray]:
        """Detect the target object referenced by the instruction.

        Updates self.target_box and self.target_depth as side effects.

        Args:
            image: RGB uint8 image (H, W, 3).
            instruction: Language instruction string.

        Returns:
            Target bbox [cx, cy, w, h] normalized, or None if nothing found.
        """
        if self.model is None:
            return None

        self._target_detection_attempts += 1

        all_boxes: List[np.ndarray] = []
        all_confs: List[float] = []

        # Query 1: full instruction
        try:
            boxes, confs, _ = self._detect_objects(image, instruction)
            for i in range(len(boxes)):
                all_boxes.append(boxes[i])
                all_confs.append(float(confs[i]))
        except Exception as e:
            print(f"detect_target: full-instruction query failed: {e}")

        # Query 2: last 2-3 words fallback
        words = instruction.lower().strip().split()
        if len(words) >= 3:
            fallback_prompt = " ".join(words[-3:]) + "."
        elif len(words) >= 1:
            fallback_prompt = " ".join(words) + "."
        else:
            fallback_prompt = None

        if fallback_prompt:
            try:
                boxes2, confs2, _ = self._detect_objects(image, fallback_prompt)
                for i in range(len(boxes2)):
                    all_boxes.append(boxes2[i])
                    all_confs.append(float(confs2[i]))
            except Exception as e:
                print(f"detect_target: fallback query failed: {e}")

        if not all_boxes:
            self._target_detection_failures += 1
            self.target_box = None
            self.target_depth = None
            return None

        best_idx = int(np.argmax(all_confs))
        best_box = np.asarray(all_boxes[best_idx], dtype=np.float32)
        best_conf = all_confs[best_idx]

        self._target_conf_sum += best_conf
        self._target_conf_count += 1
        self.target_box = best_box

        # Optional depth lookup at target center
        self.target_depth = None
        if self.depth_estimator is not None:
            try:
                pil_img = Image.fromarray(image)
                depth_output = self.depth_estimator(pil_img)
                depth_map = np.array(depth_output["depth"], dtype=np.float32)
                img_h, img_w = depth_map.shape[:2]
                px = int(np.clip(best_box[0] * img_w, 0, img_w - 1))
                py = int(np.clip(best_box[1] * img_h, 0, img_h - 1))
                d = float(depth_map[py, px])
                d_min = float(depth_map.min())
                d_max = float(depth_map.max())
                self.target_depth = (d - d_min) / (d_max - d_min + 1e-8)
            except Exception as e:
                print(f"detect_target: depth lookup failed: {e}")
                self.target_depth = None

        return best_box

    # ------------------------------------------------------------------
    # Proprio normalization
    # ------------------------------------------------------------------

    def _normalize_proprio_xyz(self, proprio: np.ndarray) -> Tuple[float, float, float]:
        """Normalize proprio XYZ to [0, 1] using workspace bounds."""
        x = (float(proprio[0]) - self.workspace_x_min) / (self.workspace_x_max - self.workspace_x_min)
        y = (float(proprio[1]) - self.workspace_y_min) / (self.workspace_y_max - self.workspace_y_min)
        z = (float(proprio[2]) - self.workspace_z_min) / (self.workspace_z_max - self.workspace_z_min)
        return (
            float(np.clip(x, 0.0, 1.0)),
            float(np.clip(y, 0.0, 1.0)),
            float(np.clip(z, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract per-frame geometric concepts.

        Reads target state from self.target_box / self.target_depth (set by
        detect_target()) and EEF state from the supplied proprio vector.

        Args:
            image: RGB uint8 image (H, W, 3). Currently only used for shape.
            instruction: Language instruction (unused here; kept for signature
                stability and potential future use).
            proprio: Optional per-frame proprio vector. If None, EEF concepts
                are set to 0.0.

        Returns:
            Dictionary of concept values (see CONCEPT_KEYS).
        """
        concepts = {k: 0.0 for k in self.CONCEPT_KEYS}

        # --- Target concepts (from stored detection) ---
        if self.target_box is not None:
            concepts["target_visible"] = 1.0
            concepts["target_cx"] = float(self.target_box[0])
            concepts["target_cy"] = float(self.target_box[1])
            concepts["target_bbox_size"] = float(self.target_box[2] * self.target_box[3])
            if self.target_depth is not None:
                concepts["target_depth"] = float(self.target_depth)

        # --- EEF concepts (from proprio) ---
        if proprio is not None and len(proprio) >= 3:
            self._frames_with_proprio += 1
            eef_x, eef_y, eef_z = self._normalize_proprio_xyz(proprio)
            concepts["eef_x"] = eef_x
            concepts["eef_y"] = eef_y
            concepts["eef_z"] = eef_z

            if self.prev_proprio is not None:
                prev_x, prev_y, prev_z = self._normalize_proprio_xyz(self.prev_proprio)
                concepts["eef_delta_x"] = eef_x - prev_x
                concepts["eef_delta_y"] = eef_y - prev_y
                concepts["eef_delta_z"] = eef_z - prev_z

            self.prev_proprio = np.asarray(proprio, dtype=np.float32).copy()
        else:
            self._frames_without_proprio += 1

        return concepts

