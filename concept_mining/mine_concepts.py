import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
import dlimp as dl
from datasets import Dataset
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

if USE_GPU:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus, "GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU enabled: {len(gpus)} device(s)")
            print(f"GPU devices: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            gpus = []
            USE_GPU = False
    else:
        print("No GPU devices found, using CPU")
        USE_GPU = False
        gpus = []
else:
    tf.config.set_visible_devices([], "GPU")
    gpus = []
    print("GPU disabled (USE_GPU=false), using CPU only")

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, StateEncoding
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from concept_mining.extractors import ProprioceptiveExtractor, GeometricExtractor

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


def extract_concepts_from_trajectory_pass_a(
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
            concept_records = extract_concepts_from_trajectory_pass_a(
                traj, dataset_name, traj_idx, extractor, use_gpu=use_gpu
            )
            if use_gpu and len(gpus) > 0:
                gpu_used_count += 1
            else:
                cpu_fallback_count += 1
        except Exception as e:
            print(f"Error extracting concepts from trajectory {traj_idx}: {e}")
            cpu_fallback_count += 1
            concept_records = extract_concepts_from_trajectory_pass_a(
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


def extract_concepts_from_trajectory_pass_b(
    traj: Dict[str, Any],
    dataset_name: str,
    episode_idx: int,
    geometric_extractor: GeometricExtractor,
) -> List[Dict[str, Any]]:
    """
    Extract Pass B (geometric) concepts from a trajectory.
    
    Returns:
        List of concept dictionaries, one per frame
    """
    concept_records = []
    
    if "observation" not in traj:
        return concept_records
    
    obs = traj["observation"]
    
    images = None
    if "image" in obs:
        images_dict = obs["image"]
        if isinstance(images_dict, dict):
            images = list(images_dict.values())[0] if images_dict else None
        else:
            images = images_dict
    elif "image_primary" in obs:
        images = obs["image_primary"]
    
    if images is None:
        return concept_records
    
    instruction = None
    if "task" in traj and "language_instruction" in traj["task"]:
        instruction = traj["task"]["language_instruction"]
        if isinstance(instruction, (bytes, tf.Tensor)):
            if isinstance(instruction, tf.Tensor):
                instruction = instruction.numpy()
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            else:
                instruction = str(instruction)
        instruction = str(instruction).strip()
    
    if not instruction:
        return concept_records
    
    if isinstance(images, tf.Tensor):
        images_np = images.numpy()
    else:
        images_np = np.array(images)
    
    if len(images_np.shape) == 4:
        traj_len = images_np.shape[0]
    elif len(images_np.shape) == 3:
        traj_len = 1
        images_np = images_np[None]
    else:
        return concept_records
    
    episode_id = f"{dataset_name}_episode_{episode_idx:06d}"
    
    for frame_idx in range(traj_len):
        img = images_np[frame_idx]
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        geometric_concepts = geometric_extractor.extract(img, instruction)
        
        concept_record = {
            "dataset_name": dataset_name,
            "episode_id": episode_id,
            "frame_index": int(frame_idx),
            "concepts": geometric_concepts,
        }
        concept_records.append(concept_record)
    
    return concept_records


def mine_concepts_pass_b(
    dataset_name: str,
    data_dir: Path,
    output_path: Path,
    train: bool = True,
    confidence_threshold: float = 0.3,
    alignment_pixel_threshold: int = 50,
    pass_a_concepts_path: Optional[Path] = None,
):
    """
    Mine Pass B (geometric) concepts from RLDS dataset using GroundingDINO.
    
    Args:
        dataset_name: Name of the RLDS dataset
        data_dir: Directory containing RLDS data
        output_path: Path to save the concept dataset
        train: Whether to use train or validation split
        confidence_threshold: GroundingDINO confidence threshold
        alignment_pixel_threshold: Pixel distance threshold for alignment
        pass_a_concepts_path: Optional path to Pass A concepts to merge
    """
    print(f"Mining Pass B (geometric) concepts from {dataset_name} (split={'train' if train else 'val'})...")
    
    oxe_dataset_name = TFDS_TO_OXE_NAME_MAP.get(dataset_name, dataset_name)
    dataset_config = OXE_DATASET_CONFIGS.get(oxe_dataset_name)
    if dataset_config is None:
        available = list(OXE_DATASET_CONFIGS.keys())[:10]
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available OXE datasets include: {', '.join(available)}... "
            f"(Total: {len(OXE_DATASET_CONFIGS)})."
        )
    
    print(f"Using OXE config name: {oxe_dataset_name}")
    
    geometric_extractor = GeometricExtractor(
        confidence_threshold=confidence_threshold,
        alignment_pixel_threshold=alignment_pixel_threshold,
    )
    
    if geometric_extractor.model is None:
        print("Error: GroundingDINO model not available. Cannot proceed with Pass B.")
        return
    
    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name=oxe_dataset_name,
        data_root_dir=data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,
        load_language=True,
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
    
    print(f"Iterating through trajectories...")
    for traj_idx, traj in enumerate(tqdm(ds.as_numpy_iterator())):
        if "observation" not in traj:
            continue
        
        concept_records = extract_concepts_from_trajectory_pass_b(
            traj, dataset_name, traj_idx, geometric_extractor
        )
        all_concept_records.extend(concept_records)
        
        if (traj_idx + 1) % 100 == 0:
            print(f"Processed {traj_idx + 1} trajectories, {len(all_concept_records)} concept records")
    
    print(f"Total concept records: {len(all_concept_records)}")
    
    if len(all_concept_records) == 0:
        print("No concept records extracted. Exiting.")
        return
    
    if pass_a_concepts_path and pass_a_concepts_path.exists():
        print(f"Merging with Pass A concepts from {pass_a_concepts_path}...")
        pass_a_dataset = load_from_disk(str(pass_a_concepts_path))
        pass_a_dict = {
            (r["episode_id"], r["frame_index"]): r["concepts"]
            for r in pass_a_dataset
        }
        
        for record in all_concept_records:
            key = (record["episode_id"], record["frame_index"])
            if key in pass_a_dict:
                record["concepts"].update(pass_a_dict[key])
    
    concept_dataset = Dataset.from_list(all_concept_records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concept_dataset.save_to_disk(str(output_path))
    
    print(f"Saved concept dataset to {output_path}")
    print(f"Dataset size: {len(concept_dataset)}")
    if len(concept_dataset) > 0:
        print(f"Sample record: {concept_dataset[0]}")


def main():
    parser = argparse.ArgumentParser(description="Mine concepts from RLDS datasets")
    parser.add_argument("--pass", type=str, choices=["a", "b", "ab"], default="a",
                       help="Which pass to run: 'a' (proprioceptive), 'b' (geometric), 'ab' (both)")
    parser.add_argument("--dataset_name", type=str, required=True, help="RLDS dataset name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing RLDS data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save concept dataset")
    parser.add_argument("--train", action="store_true", default=True, help="Use train split (default: True)")
    parser.add_argument("--val", action="store_true", help="Use validation split")
    
    parser.add_argument("--gripper_closed_threshold", type=float, default=0.05, help="Gripper closed threshold (Pass A)")
    parser.add_argument("--arm_moving_epsilon", type=float, default=0.01, help="Arm moving epsilon threshold (Pass A)")
    parser.add_argument("--table_height", type=float, default=0.0, help="Table height threshold (Pass A)")
    
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="GroundingDINO confidence threshold (Pass B)")
    parser.add_argument("--alignment_pixel_threshold", type=int, default=50, help="Alignment pixel threshold (Pass B)")
    parser.add_argument("--pass_a_concepts_path", type=str, default=None, help="Path to Pass A concepts to merge (Pass B)")
    
    args = parser.parse_args()
    
    train = not args.val
    
    if args.pass in ["a", "ab"]:
        mine_concepts_pass_a(
            dataset_name=args.dataset_name,
            data_dir=Path(args.data_dir),
            output_path=Path(args.output_path) if args.pass == "a" else Path(args.output_path) / "pass_a",
            train=train,
            gripper_closed_threshold=args.gripper_closed_threshold,
            arm_moving_epsilon=args.arm_moving_epsilon,
            table_height=args.table_height,
        )
    
    if args.pass in ["b", "ab"]:
        pass_a_path = Path(args.pass_a_concepts_path) if args.pass_a_concepts_path else (
            Path(args.output_path) / "pass_a" if args.pass == "ab" else None
        )
        mine_concepts_pass_b(
            dataset_name=args.dataset_name,
            data_dir=Path(args.data_dir),
            output_path=Path(args.output_path) if args.pass == "b" else Path(args.output_path) / "pass_b",
            train=train,
            confidence_threshold=args.confidence_threshold,
            alignment_pixel_threshold=args.alignment_pixel_threshold,
            pass_a_concepts_path=pass_a_path,
        )


if __name__ == "__main__":
    main()

