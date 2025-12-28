import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
import dlimp as dl
from datasets import Dataset
from tqdm import tqdm
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

# TensorFlow GPU setup (only affects Pass A)
# Note: Pass B (GroundingDINO) uses PyTorch and will automatically detect GPU
if USE_GPU:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus, "GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow GPU enabled: {len(gpus)} device(s)")
            print(f"TensorFlow GPU devices: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"TensorFlow GPU configuration error: {e}")
            gpus = []
            USE_GPU = False
    else:
        print("No TensorFlow GPU devices found")
        USE_GPU = False
        gpus = []
else:
    tf.config.set_visible_devices([], "GPU")
    gpus = []
    # Only print this if we're actually running Pass A
    # (We'll check pass_type in main() and print there if needed)

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, StateEncoding
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from concept_mining.extractors import ProprioceptiveExtractor, GeometricExtractor
from PIL import Image

if gpus:
    try:
        tf.config.set_visible_devices(gpus, "GPU")
    except RuntimeError:
        pass


def get_state_encoding_str(state_encoding: StateEncoding) -> str:
    mapping = {
        StateEncoding.POS_EULER: "POS_EULER",
        StateEncoding.POS_QUAT: "POS_QUAT",
        StateEncoding.JOINT: "JOINT",
        StateEncoding.NONE: "NONE",
    }
    return mapping.get(state_encoding, "NONE")


TFDS_TO_OXE_NAME_MAP = {
    "bridge": "bridge_oxe",
    "bridge_dataset": "bridge_oxe",
}


def extract_concepts_batch_gpu(
    proprio_batch: tf.Tensor,
    state_encoding_str: str,
    gripper_closed_threshold: float,
    arm_moving_epsilon: float,
    table_height: float,
) -> Dict[str, tf.Tensor]:
    """
    Extract concepts from a batch of proprioceptive states using GPU.
    
    Args:
        proprio_batch: Tensor of shape [batch_size, state_dim]
        state_encoding_str: State encoding type
        gripper_closed_threshold: Threshold for gripper closed
        arm_moving_epsilon: Threshold for arm moving
        table_height: Table height threshold
    
    Returns:
        Dictionary of concept tensors, each of shape [batch_size]
    """
    if state_encoding_str == "POS_EULER" or state_encoding_str == "POS_QUAT":
        eef_pos = proprio_batch[:, :3]
        gripper_state = proprio_batch[:, -1]
        
        gripper_closed = tf.cast(gripper_state < gripper_closed_threshold, tf.float32)
        
        eef_pos_prev = tf.concat([[eef_pos[0]], eef_pos[:-1]], axis=0)
        delta_pos = tf.norm(eef_pos - eef_pos_prev, axis=1)
        arm_moving = tf.cast(delta_pos > arm_moving_epsilon, tf.float32)
        arm_moving = tf.concat([[0.0], arm_moving[1:]], axis=0)
        
        height_above_table = tf.cast(eef_pos[:, 2] > table_height, tf.float32)
        
    elif state_encoding_str == "JOINT":
        gripper_state = proprio_batch[:, -1]
        joint_pos = proprio_batch[:, :-1]
        
        gripper_closed = tf.cast(gripper_state < gripper_closed_threshold, tf.float32)
        
        joint_pos_prev = tf.concat([[joint_pos[0]], joint_pos[:-1]], axis=0)
        delta_pos = tf.norm(joint_pos - joint_pos_prev, axis=1)
        arm_moving = tf.cast(delta_pos > arm_moving_epsilon, tf.float32)
        arm_moving = tf.concat([[0.0], arm_moving[1:]], axis=0)
        
        height_above_table = tf.zeros_like(gripper_closed)
    else:
        batch_size = tf.shape(proprio_batch)[0]
        gripper_closed = tf.zeros([batch_size], dtype=tf.float32)
        arm_moving = tf.zeros([batch_size], dtype=tf.float32)
        height_above_table = tf.zeros([batch_size], dtype=tf.float32)
    
    return {
        "gripper_closed": gripper_closed,
        "arm_moving": arm_moving,
        "height_above_table": height_above_table,
    }


def extract_concepts_from_trajectory(
    traj: Dict[str, Any],
    dataset_name: str,
    episode_idx: int,
    extractor: ProprioceptiveExtractor,
    use_gpu: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract concepts from a single trajectory.
    
    Returns:
        List of concept dictionaries, one per frame
    """
    concept_records = []
    
    if "observation" not in traj or "proprio" not in traj["observation"]:
        return concept_records
    
    dataset_config = OXE_DATASET_CONFIGS.get(dataset_name, {})
    state_encoding = dataset_config.get("state_encoding", StateEncoding.NONE)
    state_encoding_str = get_state_encoding_str(state_encoding)
    
    if state_encoding == StateEncoding.NONE:
        return concept_records
    
    proprio = traj["observation"]["proprio"]
    
    if isinstance(proprio, tf.Tensor):
        traj_len = int(proprio.shape[0])
        proprio_tf = proprio
    else:
        traj_len = len(proprio)
        proprio_tf = tf.constant(proprio, dtype=tf.float32)
    
    episode_id = f"{dataset_name}_episode_{episode_idx:06d}"
    
    if use_gpu and gpus:
        try:
            with tf.device("/GPU:0"):
                concepts_dict = extract_concepts_batch_gpu(
                    proprio_tf,
                    state_encoding_str,
                    extractor.gripper_closed_threshold,
                    extractor.arm_moving_epsilon,
                    extractor.table_height,
                )
                
                concepts_np = {k: v.numpy() for k, v in concepts_dict.items()}
                
                for frame_idx in range(traj_len):
                    concept_record = {
                        "dataset_name": dataset_name,
                        "episode_id": episode_id,
                        "frame_index": int(frame_idx),
                        "concepts": {
                            "gripper_closed": float(concepts_np["gripper_closed"][frame_idx]),
                            "arm_moving": float(concepts_np["arm_moving"][frame_idx]),
                            "height_above_table": float(concepts_np["height_above_table"][frame_idx]),
                        },
                    }
                    concept_records.append(concept_record)
        except Exception as e:
            print(f"GPU extraction failed, falling back to CPU: {e}")
            use_gpu = False
    
    if not use_gpu or not gpus:
        extractor.reset()
        proprio_np = proprio_tf.numpy() if isinstance(proprio_tf, tf.Tensor) else np.array(proprio)
        
        for frame_idx in range(traj_len):
            proprio_frame = proprio_np[frame_idx]
            concepts = extractor.extract(proprio_frame, state_encoding_str)
            
            concept_record = {
                "dataset_name": dataset_name,
                "episode_id": episode_id,
                "frame_index": int(frame_idx),
                "concepts": concepts,
            }
            concept_records.append(concept_record)
    
    return concept_records


def mine_concepts_pass_a(
    dataset_name: str,
    data_dir: Path,
    output_path: Path,
    train: bool = True,
    gripper_closed_threshold: float = 0.05,
    arm_moving_epsilon: float = 0.01,
    table_height: float = 0.0,
):
    """
    Mine Pass A (proprioceptive) concepts from RLDS dataset.
    
    Args:
        dataset_name: Name of the RLDS dataset
        data_dir: Directory containing RLDS data
        output_path: Path to save the concept dataset
        train: Whether to use train or validation split
        gripper_closed_threshold: Threshold for gripper closed concept
        arm_moving_epsilon: Threshold for arm moving concept
        table_height: Height threshold for height_above_table concept
    """
    print(f"Mining concepts from {dataset_name} (split={'train' if train else 'val'})...")
    
    oxe_dataset_name = TFDS_TO_OXE_NAME_MAP.get(dataset_name, dataset_name)
    dataset_config = OXE_DATASET_CONFIGS.get(oxe_dataset_name)
    if dataset_config is None:
        available = list(OXE_DATASET_CONFIGS.keys())[:10]
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available OXE datasets include: {', '.join(available)}... "
            f"(Total: {len(OXE_DATASET_CONFIGS)}). "
            f"Note: TFDS dataset names may differ from OXE config names."
        )
    
    print(f"Using OXE config name: {oxe_dataset_name}")
    
    state_encoding = dataset_config.get("state_encoding", StateEncoding.NONE)
    if state_encoding == StateEncoding.NONE:
        print(f"Warning: Dataset {dataset_name} has no state encoding. Skipping.")
        return
    
    extractor = ProprioceptiveExtractor(
        gripper_closed_threshold=gripper_closed_threshold,
        arm_moving_epsilon=arm_moving_epsilon,
        table_height=table_height,
    )
    
    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name=oxe_dataset_name,
        data_root_dir=data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=True,
        load_language=False,
    )
    
    dataset_kwargs["name"] = dataset_name
    
    try:
        ds, _ = make_dataset_from_rlds(
            train=train,
            shuffle=False,
            num_parallel_reads=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            **dataset_kwargs,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    all_concept_records = []
    
    use_gpu = USE_GPU and len(gpus) > 0
    print(f"Iterating through trajectories (GPU: {use_gpu})...")
    if not use_gpu:
        print("Note: Running in CPU-only mode. Set USE_GPU=true environment variable to enable GPU.")
    
    gpu_used_count = 0
    cpu_fallback_count = 0
    
    for traj_idx, traj in enumerate(tqdm(ds.as_numpy_iterator())):
        if "observation" not in traj:
            continue
        
        if "proprio" not in traj["observation"]:
            continue
        
        try:
            concept_records = extract_concepts_from_trajectory(
                traj, dataset_name, traj_idx, extractor, use_gpu=use_gpu
            )
            if use_gpu and len(gpus) > 0:
                gpu_used_count += 1
            else:
                cpu_fallback_count += 1
        except Exception as e:
            print(f"Error extracting concepts from trajectory {traj_idx}: {e}")
            cpu_fallback_count += 1
            concept_records = extract_concepts_from_trajectory(
                traj, dataset_name, traj_idx, extractor, use_gpu=False
            )
        
        all_concept_records.extend(concept_records)
        
        if (traj_idx + 1) % 100 == 0:
            print(f"Processed {traj_idx + 1} trajectories, {len(all_concept_records)} concept records")
            if use_gpu:
                print(f"  GPU trajectories: {gpu_used_count}, CPU fallback: {cpu_fallback_count}")
    
    print(f"Total concept records: {len(all_concept_records)}")
    
    if len(all_concept_records) == 0:
        print("No concept records extracted. Exiting.")
        return
    
    concept_dataset = Dataset.from_list(all_concept_records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concept_dataset.save_to_disk(str(output_path))
    
    print(f"Saved concept dataset to {output_path}")
    print(f"Dataset size: {len(concept_dataset)}")
    if len(concept_dataset) > 0:
        print(f"Sample record: {concept_dataset[0]}")


def mine_concepts_pass_b(
    dataset_name: str,
    data_dir: Path,
    output_path: Path,
    train: bool = True,
    groundingdino_config: Optional[str] = None,
    groundingdino_checkpoint: Optional[str] = None,
    text_threshold: float = 0.25,
    box_threshold: float = 0.3,
    alignment_pixel_threshold: float = 50.0,
):
    """
    Mine Pass B (geometric) concepts from RLDS dataset using GroundingDINO.
    
    Args:
        dataset_name: Name of the RLDS dataset
        data_dir: Directory containing RLDS data
        output_path: Path to save the concept dataset
        train: Whether to use train or validation split
        groundingdino_config: Path to GroundingDINO config file
        groundingdino_checkpoint: Path to GroundingDINO checkpoint
        text_threshold: Threshold for text matching confidence
        box_threshold: Threshold for bounding box confidence
        alignment_pixel_threshold: Pixel distance threshold for alignment
    """
    print(f"Mining Pass B (geometric) concepts from {dataset_name} (split={'train' if train else 'val'})...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"PyTorch GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("PyTorch GPU not available, using CPU")
    
    if not groundingdino_config or not groundingdino_checkpoint:
        print("Error: GroundingDINO config and checkpoint paths are required for Pass B")
        print("Install GroundingDINO: git clone https://github.com/IDEA-Research/GroundingDINO")
        print("Download checkpoint: wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth")
        return
    
    oxe_dataset_name = TFDS_TO_OXE_NAME_MAP.get(dataset_name, dataset_name)
    dataset_config = OXE_DATASET_CONFIGS.get(oxe_dataset_name)
    if dataset_config is None:
        available = list(OXE_DATASET_CONFIGS.keys())[:10]
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available OXE datasets include: {', '.join(available)}... "
        )
    
    extractor = GeometricExtractor(
        model_config_path=groundingdino_config,
        model_checkpoint_path=groundingdino_checkpoint,
        text_threshold=text_threshold,
        box_threshold=box_threshold,
        alignment_pixel_threshold=alignment_pixel_threshold,
        device=device,
    )
    
    if extractor.model is None:
        print("Failed to load GroundingDINO model. Exiting.")
        return
    
    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name=oxe_dataset_name,
        data_root_dir=data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=True,
        load_language=True,
    )
    
    dataset_kwargs["name"] = dataset_name
    
    try:
        ds, _ = make_dataset_from_rlds(
            train=train,
            shuffle=False,
            num_parallel_reads=1,
            num_parallel_calls=1,
            **dataset_kwargs,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    all_concept_records = []
    
    for traj_idx, traj in enumerate(tqdm(ds.as_numpy_iterator())):
        if "observation" not in traj:
            continue
        
        if "proprio" not in traj["observation"]:
            continue
        
        if "task" not in traj or "language_instruction" not in traj["task"]:
            continue
        
        instruction = traj["task"]["language_instruction"]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        elif isinstance(instruction, np.ndarray):
            instruction = str(instruction[0]) if len(instruction) > 0 else ""
        
        proprio = traj["observation"]["proprio"]
        if isinstance(proprio, tf.Tensor):
            proprio = proprio.numpy()
        
        num_frames = len(proprio) if len(proprio.shape) > 1 else 1
        
        episode_id = f"{dataset_name}_episode_{traj_idx:06d}"
        
        for frame_idx in range(num_frames):
            proprio_frame = proprio[frame_idx] if len(proprio.shape) > 1 else proprio
            
            img = extract_image_from_trajectory(traj, frame_idx)
            if img is None:
                continue
            
            concepts = extractor.extract(
                image=img,
                instruction=instruction,
                proprio=proprio_frame,
            )
            
            concept_record = {
                "dataset_name": dataset_name,
                "episode_id": episode_id,
                "frame_index": int(frame_idx),
                "concepts": concepts,
            }
            all_concept_records.append(concept_record)
        
        if (traj_idx + 1) % 10 == 0:
            print(f"Processed {traj_idx + 1} trajectories, {len(all_concept_records)} concept records")
    
    print(f"Total concept records: {len(all_concept_records)}")
    
    if len(all_concept_records) == 0:
        print("No concept records extracted. Exiting.")
        return
    
    concept_dataset = Dataset.from_list(all_concept_records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concept_dataset.save_to_disk(str(output_path))
    
    print(f"Saved concept dataset to {output_path}")
    print(f"Dataset size: {len(concept_dataset)}")
    if len(concept_dataset) > 0:
        print(f"Sample record: {concept_dataset[0]}")


def extract_image_from_trajectory(traj, frame_idx):
    """Extract image from trajectory at specific frame index."""
    if "observation" not in traj:
        return None
    
    obs = traj["observation"]
    
    img = None
    if "image_primary" in obs:
        img = obs["image_primary"]
    elif "image" in obs:
        images = obs["image"]
        if isinstance(images, dict):
            img = images.get("primary") or list(images.values())[0]
        else:
            img = images
    elif "rgb" in obs:
        img = obs["rgb"]
    
    if img is None:
        return None
    
    if isinstance(img, tf.Tensor):
        img_np = img.numpy()
    else:
        img_np = np.array(img)
    
    if len(img_np.shape) == 4:
        img_np = img_np[frame_idx]
    elif len(img_np.shape) == 3:
        img_np = img_np
    else:
        return None
    
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    return Image.fromarray(img_np)


def main():
    parser = argparse.ArgumentParser(description="Mine concepts from RLDS datasets")
    parser.add_argument("--pass_type", type=str, choices=["a", "b", "ab"], default="a",
                       help="Which pass to run: 'a' (proprioceptive), 'b' (geometric), 'ab' (both)")
    parser.add_argument("--dataset_name", type=str, required=True, help="RLDS dataset name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing RLDS data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save concept dataset")
    parser.add_argument("--train", action="store_true", default=True, help="Use train split (default: True)")
    parser.add_argument("--val", action="store_true", help="Use validation split")
    
    parser.add_argument("--gripper_closed_threshold", type=float, default=0.05, help="Gripper closed threshold (Pass A)")
    parser.add_argument("--arm_moving_epsilon", type=float, default=0.01, help="Arm moving epsilon threshold (Pass A)")
    parser.add_argument("--table_height", type=float, default=0.0, help="Table height threshold (Pass A)")
    
    parser.add_argument("--groundingdino_config", type=str, default=None,
                       help="Path to GroundingDINO config file (Pass B)")
    parser.add_argument("--groundingdino_checkpoint", type=str, default=None,
                       help="Path to GroundingDINO checkpoint (Pass B)")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for GroundingDINO (Pass B)")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box threshold for GroundingDINO (Pass B)")
    parser.add_argument("--alignment_pixel_threshold", type=float, default=50.0,
                       help="Pixel distance threshold for alignment (Pass B)")
    
    args = parser.parse_args()
    
    train = not args.val
    
    if args.pass_type in ["a", "ab"]:
        output_path_a = Path(args.output_path) if args.pass_type == "a" else Path(args.output_path).with_suffix(".pass_a")
        mine_concepts_pass_a(
            dataset_name=args.dataset_name,
            data_dir=Path(args.data_dir),
            output_path=output_path_a,
            train=train,
            gripper_closed_threshold=args.gripper_closed_threshold,
            arm_moving_epsilon=args.arm_moving_epsilon,
            table_height=args.table_height,
        )
    
    if args.pass_type in ["b", "ab"]:
        output_path_b = Path(args.output_path) if args.pass_type == "b" else Path(args.output_path).with_suffix(".pass_b")
        mine_concepts_pass_b(
            dataset_name=args.dataset_name,
            data_dir=Path(args.data_dir),
            output_path=output_path_b,
            train=train,
            groundingdino_config=args.groundingdino_config,
            groundingdino_checkpoint=args.groundingdino_checkpoint,
            text_threshold=args.text_threshold,
            box_threshold=args.box_threshold,
            alignment_pixel_threshold=args.alignment_pixel_threshold,
        )


if __name__ == "__main__":
    main()

