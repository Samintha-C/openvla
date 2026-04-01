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
    """Extract geometric concepts using GroundingDINO.

    Produces a fixed-dimension vector of hybrid continuous + categorical
    spatial concepts per frame, designed for downstream consumption by a
    linear concept-bottleneck layer.  Concepts follow the SCoBots design:
    signed per-axis directed distances (never Euclidean), temporal deltas,
    and overlap/size proxies.
    """

    # Keys returned by extract(), in deterministic order.
    CONCEPT_KEYS: List[str] = [
        "target_visible",
        "obstacle_present",
        "dx_gripper_target",
        "dy_gripper_target",
        "target_relative_size",
        "gripper_target_iou",
        "target_dx_delta",
        "target_dy_delta",
    ]

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        alignment_pixel_threshold: int = 50,
        device: Optional[str] = None,
        detection_stride: int = 1,
        gripper_query: str = "robot gripper . robot arm . end effector .",
        obstacle_query: str = "object . item . container . tool .",
    ):
        """
        Args:
            confidence_threshold: Minimum confidence for object detection.
            alignment_pixel_threshold: Kept for backward compat (unused).
            device: Device to run GroundingDINO on.
            detection_stride: Run detection every N frames; carry forward
                for skipped frames.
            gripper_query: GroundingDINO text prompt for gripper detection.
            obstacle_query: GroundingDINO text prompt for obstacle detection.
        """
        self.confidence_threshold = confidence_threshold
        self.alignment_pixel_threshold = alignment_pixel_threshold
        if device is None:
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.detection_stride = max(1, detection_stride)
        self.gripper_query = gripper_query
        self.obstacle_query = obstacle_query

        self.model = None
        self._load_model()
        self.reset()

    def reset(self) -> None:
        """Clear all per-trajectory temporal state. Call between trajectories."""
        self.prev_target_cx: Optional[float] = None
        self.prev_target_cy: Optional[float] = None
        self._cached_gripper_box: Optional[np.ndarray] = None
        self._frame_counter: int = 0
        self._last_detection_result: Optional[Dict[str, float]] = None

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
    # Helper: IoU in normalised cxcywh
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes in [cx, cy, w, h] normalised format."""
        # Convert to x1y1x2y2
        x1_a, y1_a = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
        x2_a, y2_a = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
        x1_b, y1_b = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
        x2_b, y2_b = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)

        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = box1[2] * box1[3]
        area_b = box2[2] * box2[3]
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area / union)

    # ------------------------------------------------------------------
    # Helper: raw GroundingDINO detection
    # ------------------------------------------------------------------

    def _detect_objects(
        self, image: np.ndarray, text_prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Run GroundingDINO detection.

        Args:
            image: RGB uint8 (H, W, 3).
            text_prompt: Caption/query for GroundingDINO.

        Returns:
            (boxes, confidences, phrases) where boxes is (N, 4) normalised
            cxcywh, confidences is (N,), phrases is length-N list of strings.
            Returns empty arrays when nothing is detected.
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
    # Helper: gripper detection (cached per trajectory)
    # ------------------------------------------------------------------

    def _detect_gripper(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect gripper. Returns [cx, cy, w, h] normalised or None."""
        if self._cached_gripper_box is not None:
            return self._cached_gripper_box

        boxes, confs, _ = self._detect_objects(image, self.gripper_query)
        if len(boxes) > 0:
            best = boxes[confs.argmax()]
            self._cached_gripper_box = best
            return best
        return None

    # ------------------------------------------------------------------
    # Helper: target detection (highest-conf non-gripper detection)
    # ------------------------------------------------------------------

    def _detect_target(
        self,
        image: np.ndarray,
        instruction: str,
        gripper_box: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Detect target object. Returns [cx, cy, w, h] normalised or None.

        Uses two GroundingDINO queries — the full instruction and a
        fallback noun-phrase extraction — then picks the highest-confidence
        detection that does not overlap significantly with the gripper.
        """
        all_boxes: List[np.ndarray] = []
        all_confs: List[float] = []

        # Query 1: full instruction
        boxes, confs, _ = self._detect_objects(image, instruction)
        for i in range(len(boxes)):
            all_boxes.append(boxes[i])
            all_confs.append(float(confs[i]))

        # Query 2: fallback last noun phrase (last 2-3 words)
        words = instruction.lower().strip().split()
        if len(words) >= 3:
            fallback_prompt = " ".join(words[-3:]) + "."
        elif len(words) >= 1:
            fallback_prompt = " ".join(words) + "."
        else:
            fallback_prompt = None

        if fallback_prompt:
            boxes2, confs2, _ = self._detect_objects(image, fallback_prompt)
            for i in range(len(boxes2)):
                all_boxes.append(boxes2[i])
                all_confs.append(float(confs2[i]))

        if not all_boxes:
            return None

        # Sort by confidence descending, pick best non-gripper box
        indices = sorted(range(len(all_confs)), key=lambda i: all_confs[i], reverse=True)
        for idx in indices:
            box = all_boxes[idx]
            if gripper_box is not None and self._compute_iou(box, gripper_box) > 0.3:
                continue
            return box
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        image: np.ndarray,
        instruction: str,
        gripper_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, float]:
        """Extract geometric concepts from a single frame.

        Args:
            image: RGB image as numpy array (H, W, 3) uint8.
            instruction: Language instruction for the current trajectory.
            gripper_bbox: Optional gripper bounding box (x1, y1, x2, y2) in
                pixels. If None, the gripper is auto-detected or a fixed
                prior is used.

        Returns:
            Dictionary of concept values (see CONCEPT_KEYS).
        """
        concepts = {k: 0.0 for k in self.CONCEPT_KEYS}

        if self.model is None:
            return concepts

        # --- stride: carry forward previous detection on skipped frames ---
        self._frame_counter += 1
        if (self._frame_counter - 1) % self.detection_stride != 0:
            if self._last_detection_result is not None:
                return dict(self._last_detection_result)
            return concepts

        img_h, img_w = image.shape[:2]

        try:
            # --- Gripper box (normalised cxcywh) ---
            if gripper_bbox is not None:
                # Convert pixel xyxy to normalised cxcywh
                x1, y1, x2, y2 = gripper_bbox
                gripper_box = np.array([
                    (x1 + x2) / 2.0 / img_w,
                    (y1 + y2) / 2.0 / img_h,
                    (x2 - x1) / img_w,
                    (y2 - y1) / img_h,
                ])
            else:
                gripper_box = self._detect_gripper(image)
                if gripper_box is None:
                    # Fixed prior: center-bottom of BridgeDataV2 frame
                    gripper_box = np.array([0.5, 0.75, 0.15, 0.15])

            # --- Target detection ---
            target_box = self._detect_target(image, instruction, gripper_box)

            if target_box is not None:
                concepts["target_visible"] = 1.0
                target_cx, target_cy = float(target_box[0]), float(target_box[1])
                target_w, target_h = float(target_box[2]), float(target_box[3])
                gripper_cx, gripper_cy = float(gripper_box[0]), float(gripper_box[1])

                # Signed per-axis directed distance (normalised by image dims,
                # already in [0,1] space so range is ~ [-1, 1])
                concepts["dx_gripper_target"] = target_cx - gripper_cx
                concepts["dy_gripper_target"] = target_cy - gripper_cy

                # Size / depth proxy
                concepts["target_relative_size"] = target_w * target_h

                # Overlap / grasp proxy
                concepts["gripper_target_iou"] = self._compute_iou(target_box, gripper_box)

                # Temporal deltas
                if self.prev_target_cx is not None:
                    concepts["target_dx_delta"] = target_cx - self.prev_target_cx
                    concepts["target_dy_delta"] = target_cy - self.prev_target_cy

                self.prev_target_cx = target_cx
                self.prev_target_cy = target_cy
            else:
                # Target not visible — don't update temporal state so that
                # the next visible frame computes delta from the last
                # known position.
                pass

            # --- Obstacle detection ---
            obs_boxes, obs_confs, _ = self._detect_objects(image, self.obstacle_query)
            for i in range(len(obs_boxes)):
                box = obs_boxes[i]
                if target_box is not None and self._compute_iou(box, target_box) > 0.3:
                    continue
                if self._compute_iou(box, gripper_box) > 0.3:
                    continue
                concepts["obstacle_present"] = 1.0
                break

        except Exception as e:
            print(f"Error in geometric extraction: {e}")

        self._last_detection_result = dict(concepts)
        return concepts

