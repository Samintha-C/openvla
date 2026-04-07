import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
import dlimp as dl
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from PIL import Image as PILImage

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


def _log_traj_structure(traj: Dict[str, Any], traj_idx: int) -> None:
    """Log the structure of the first trajectory for debugging."""
    print(f"\n{'='*60}")
    print(f"[DEBUG] Trajectory {traj_idx} structure dump:")
    print(f"  Top-level keys: {list(traj.keys())}")
    if "observation" in traj:
        obs = traj["observation"]
        print(f"  observation keys: {list(obs.keys())}")
        for k, v in obs.items():
            if hasattr(v, 'shape'):
                print(f"    obs[{k!r}]: shape={v.shape}, dtype={v.dtype}")
            elif hasattr(v, '__len__'):
                print(f"    obs[{k!r}]: type={type(v).__name__}, len={len(v)}")
            else:
                print(f"    obs[{k!r}]: type={type(v).__name__}, value={repr(v)[:100]}")
    if "task" in traj:
        task = traj["task"]
        print(f"  task keys: {list(task.keys())}")
        for k, v in task.items():
            print(f"    task[{k!r}]: type={type(v).__name__}, value={repr(v)[:120]}")
    if "action" in traj:
        a = traj["action"]
        if hasattr(a, 'shape'):
            print(f"  action: shape={a.shape}, dtype={a.dtype}")
    print(f"{'='*60}\n")


def extract_concepts_from_trajectory_pass_b(
    traj: Dict[str, Any],
    dataset_name: str,
    episode_idx: int,
    geometric_extractor: GeometricExtractor,
    verbose: bool = False,
    images_save_dir: Optional[Path] = None,
    gripper_closed_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Extract Pass B (geometric) concepts from a trajectory.

    Runs GroundingDINO once on the first frame to detect the target, then
    re-runs it when the gripper state flips (a proxy for keyframes where the
    target may have moved). EEF concepts come from per-frame proprio.
    """
    concept_records: List[Dict[str, Any]] = []
    geometric_extractor.reset()

    if "observation" not in traj:
        if verbose:
            print(f"  [traj {episode_idx}] SKIP: no 'observation' key")
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
        if verbose:
            print(f"  [traj {episode_idx}] SKIP: no images found. obs keys: {list(obs.keys())}")
        return concept_records

    # Proprio may be absent for some datasets — degrade gracefully.
    proprio_np: Optional[np.ndarray] = None
    if "proprio" in obs:
        proprio_raw = obs["proprio"]
        if isinstance(proprio_raw, tf.Tensor):
            proprio_np = proprio_raw.numpy()
        else:
            proprio_np = np.asarray(proprio_raw)
    else:
        if verbose:
            print(f"  [traj {episode_idx}] WARNING: no proprio in obs; EEF concepts will be zero.")

    instruction = None
    if "task" in traj and "language_instruction" in traj["task"]:
        instruction = traj["task"]["language_instruction"]
        if isinstance(instruction, tf.Tensor):
            instruction = instruction.numpy()
        if isinstance(instruction, np.ndarray):
            instruction = instruction.flat[0] if instruction.size > 0 else b""
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if len(instruction) > 0 else b""
        if isinstance(instruction, (bytes, np.bytes_)):
            instruction = instruction.decode("utf-8", errors="replace")
        instruction = str(instruction).strip()

    if not instruction:
        if verbose:
            traj_keys = list(traj.keys())
            task_keys = list(traj.get("task", {}).keys()) if isinstance(traj.get("task"), dict) else "N/A"
            print(f"  [traj {episode_idx}] SKIP: no instruction. traj keys: {traj_keys}, task keys: {task_keys}")
        return concept_records

    if isinstance(images, tf.Tensor):
        images_np = images.numpy()
    else:
        images_np = np.array(images)

    if verbose:
        print(f"  [traj {episode_idx}] Raw images: shape={images_np.shape}, dtype={images_np.dtype}")

    # RLDS stores images as encoded byte strings (JPEG/PNG). Decode them.
    if images_np.dtype.kind in ("S", "U", "O"):
        from PIL import Image as PILImage
        import io
        if verbose:
            sample = images_np.flat[0] if images_np.size > 0 else None
            print(f"  [traj {episode_idx}] Decoding byte-string images. sample type={type(sample).__name__}, len={len(sample) if sample else 0}")
        decoded = []
        for raw in images_np:
            if isinstance(raw, (bytes, np.bytes_)):
                decoded.append(np.array(PILImage.open(io.BytesIO(raw)).convert("RGB")))
        if not decoded:
            if verbose:
                print(f"  [traj {episode_idx}] SKIP: no decodable images")
            return concept_records
        images_np = np.stack(decoded)
        if verbose:
            print(f"  [traj {episode_idx}] Decoded images: shape={images_np.shape}, dtype={images_np.dtype}")

    if len(images_np.shape) == 4:
        traj_len = images_np.shape[0]
    elif len(images_np.shape) == 3:
        traj_len = 1
        images_np = images_np[None]
    else:
        if verbose:
            print(f"  [traj {episode_idx}] SKIP: unexpected image shape {images_np.shape}")
        return concept_records

    # Align proprio length to image length (some datasets may be off by one)
    if proprio_np is not None and len(proprio_np) < traj_len:
        if verbose:
            print(f"  [traj {episode_idx}] WARNING: proprio len {len(proprio_np)} < traj len {traj_len}, padding with last value")
        pad = np.repeat(proprio_np[-1:], traj_len - len(proprio_np), axis=0)
        proprio_np = np.concatenate([proprio_np, pad], axis=0)

    episode_id = f"{dataset_name}_episode_{episode_idx:06d}"

    def _prepare(img):
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                return (img * 255).astype(np.uint8)
            return img.astype(np.uint8)
        return img

    # --- Run initial target detection on the first frame ---
    first_img = _prepare(images_np[0])
    geometric_extractor.detect_target(first_img, instruction)

    # If no proprio is available, re-detect every frame as a fallback
    # (otherwise target concepts would be stale and EEF concepts are zero,
    # leaving nothing time-varying).
    redetect_every_frame = proprio_np is None

    prev_gripper_closed: Optional[bool] = None

    for frame_idx in range(traj_len):
        img = _prepare(images_np[frame_idx])
        proprio_frame = proprio_np[frame_idx] if proprio_np is not None else None

        # Re-run target detection on gripper-state changes (keyframe proxy)
        if proprio_frame is not None:
            gripper_val = float(proprio_frame[-1])
            gripper_closed = gripper_val < gripper_closed_threshold
            if prev_gripper_closed is not None and gripper_closed != prev_gripper_closed:
                geometric_extractor.detect_target(img, instruction)
            prev_gripper_closed = gripper_closed
        elif redetect_every_frame and frame_idx > 0:
            geometric_extractor.detect_target(img, instruction)

        geometric_concepts = geometric_extractor.extract(img, instruction, proprio_frame)

        concept_record = {
            "dataset_name": dataset_name,
            "episode_id": episode_id,
            "frame_index": int(frame_idx),
            "instruction": instruction,
            "concepts": geometric_concepts,
        }
        concept_records.append(concept_record)

        if images_save_dir is not None:
            img_filename = f"{episode_id}_frame_{frame_idx:05d}.jpg"
            PILImage.fromarray(img).save(images_save_dir / img_filename, quality=85)

    return concept_records


def mine_concepts_pass_b(
    dataset_name: str,
    data_dir: Path,
    output_path: Path,
    train: bool = True,
    confidence_threshold: float = 0.3,
    pass_a_concepts_path: Optional[Path] = None,
    max_trajectories: Optional[int] = None,
    save_images: bool = False,
    use_depth: bool = False,
    depth_model: str = "LiheYoung/depth-anything-v2-small-hf",
    workspace_x_min: float = 0.1,
    workspace_x_max: float = 0.4,
    workspace_y_min: float = -0.15,
    workspace_y_max: float = 0.25,
    workspace_z_min: float = 0.0,
    workspace_z_max: float = 0.3,
    gripper_closed_threshold: float = 0.05,
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
        detection_stride: Run GroundingDINO every N frames, carry forward for skipped frames
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
        use_depth=use_depth,
        depth_model_name=depth_model,
        workspace_x_min=workspace_x_min,
        workspace_x_max=workspace_x_max,
        workspace_y_min=workspace_y_min,
        workspace_y_max=workspace_y_max,
        workspace_z_min=workspace_z_min,
        workspace_z_max=workspace_z_max,
    )
    print(f"Target detection: once per trajectory + on gripper-state changes")
    print(f"Depth estimation: {'enabled' if use_depth else 'disabled'}")
    if max_trajectories is not None:
        print(f"Max trajectories: {max_trajectories}")

    images_save_dir = None
    if save_images:
        images_save_dir = output_path / "images"
        images_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving images to {images_save_dir}")

    if geometric_extractor.model is None:
        print("Error: GroundingDINO model not available. Cannot proceed with Pass B.")
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
    geometric_extractor.reset_stats()

    import sys
    import time

    print(f"Iterating through trajectories...", flush=True)
    skip_count = 0
    skip_reasons = {"no_observation": 0, "no_records": 0, "exception": 0}
    concept_sums: dict = {}
    concept_counts: dict = {}
    instruction_sample: list = []
    first_traj_logged = False
    t_start = time.time()

    for traj_idx, traj in enumerate(tqdm(ds.as_numpy_iterator(), file=sys.stderr)):
        # Log structure of the very first trajectory for debugging
        if not first_traj_logged:
            _log_traj_structure(traj, traj_idx)
            sys.stdout.flush()
            first_traj_logged = True
            t_first = time.time()

        if "observation" not in traj:
            skip_reasons["no_observation"] += 1
            continue

        # Verbose for first 3 trajectories to diagnose data flow
        verbose = True

        try:
            concept_records = extract_concepts_from_trajectory_pass_b(
                traj, dataset_name, traj_idx, geometric_extractor,
                verbose=verbose, images_save_dir=images_save_dir,
                gripper_closed_threshold=gripper_closed_threshold,
            )
        except Exception as e:
            print(f"Error extracting concepts from trajectory {traj_idx}, skipping: {e}")
            import traceback
            traceback.print_exc()
            skip_reasons["exception"] += 1
            skip_count += 1
            continue

        if traj_idx == 0:
            t_first_done = time.time()
            print(f"\n[TIMING] First trajectory: {t_first_done - t_first:.1f}s, {len(concept_records)} records", flush=True)
            if concept_records:
                print(f"[SAMPLE] {concept_records[0]['concepts']}", flush=True)

        if not concept_records:
            skip_reasons["no_records"] += 1

        # Collect instruction sample from first 10 trajectories
        if len(instruction_sample) < 10 and concept_records:
            instr = concept_records[0].get("instruction", "")
            if instr and instr not in instruction_sample:
                instruction_sample.append(instr)
                if len(instruction_sample) == 1:
                    print(f"\n[INSTRUCTIONS sample (first 10 unique)]", flush=True)
                print(f"  {len(instruction_sample):2d}. {instr}", flush=True)

        # Track per-concept stats
        concept_nonzero: dict = concept_sums.get("__nonzero__", {})
        for rec in concept_records:
            for k, v in rec["concepts"].items():
                concept_sums[k] = concept_sums.get(k, 0.0) + v
                concept_counts[k] = concept_counts.get(k, 0) + 1
                if v > 1e-6:
                    concept_nonzero[k] = concept_nonzero.get(k, 0) + 1
        concept_sums["__nonzero__"] = concept_nonzero

        all_concept_records.extend(concept_records)

        if max_trajectories is not None and traj_idx + 1 >= max_trajectories:
            print(f"\n[LIMIT] Reached max_trajectories={max_trajectories}, stopping early.", flush=True)
            break

        if (traj_idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (traj_idx + 1) / elapsed
            frames = len(all_concept_records)
            gstats = geometric_extractor.get_stats()
            print(f"\n[{traj_idx+1} trajs | {elapsed:.0f}s | {rate:.1f} traj/s | frames={frames} | skipped={skip_count}]", flush=True)
            print(f"  Target:  attempts={gstats['target_detection_attempts']}  failures={gstats['target_detection_failures']} ({gstats['target_detection_fail_pct']:.1f}%)  mean_conf={gstats['target_mean_conf']:.3f}", flush=True)
            print(f"  Proprio: with={gstats['frames_with_proprio']}  without={gstats['frames_without_proprio']}", flush=True)
            concept_nonzero = concept_sums.get("__nonzero__", {})
            print(f"  Concept rates (positive or nonzero):", flush=True)
            for k in concept_counts:
                n = concept_counts[k]
                mean_val = concept_sums[k] / n if n > 0 else 0.0
                pos_rate = concept_nonzero.get(k, 0) / n if n > 0 else 0.0
                print(f"    {k:<28} mean={mean_val:+.4f}  nonzero={pos_rate:.1%}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Pass B mining complete in {elapsed_total:.0f}s")
    print(f"  Total trajectories processed: {traj_idx + 1}")
    print(f"  Total concept records: {len(all_concept_records)}")
    print(f"  Skipped: {skip_count} (reasons: {skip_reasons})")
    if len(all_concept_records) > 0:
        target_visible_count = concept_sums.get("__nonzero__", {}).get("target_visible", 0)
        tv_pct = 100.0 * target_visible_count / len(all_concept_records)
        print(f"  Target visible in {target_visible_count}/{len(all_concept_records)} frames ({tv_pct:.1f}%)")
    print(f"{'='*60}")
    
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
    parser.add_argument("--pass", type=str, choices=["a", "b", "ab"], default="a", dest="pass_type",
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
    parser.add_argument("--pass_a_concepts_path", type=str, default=None, help="Path to Pass A concepts to merge (Pass B)")
    parser.add_argument("--max_trajectories", type=int, default=None, help="Stop after N trajectories (Pass B, default: None = all)")
    parser.add_argument("--save_images", action="store_true", help="Save each frame as JPEG alongside the concept dataset (Pass B)")
    parser.add_argument("--use_depth", action="store_true", help="Enable monocular depth estimation (Pass B)")
    parser.add_argument("--depth_model", type=str, default="LiheYoung/depth-anything-v2-small-hf", help="HF model name for depth (Pass B)")
    parser.add_argument("--workspace_x_min", type=float, default=0.1, help="Workspace X lower bound (Pass B)")
    parser.add_argument("--workspace_x_max", type=float, default=0.4, help="Workspace X upper bound (Pass B)")
    parser.add_argument("--workspace_y_min", type=float, default=-0.15, help="Workspace Y lower bound (Pass B)")
    parser.add_argument("--workspace_y_max", type=float, default=0.25, help="Workspace Y upper bound (Pass B)")
    parser.add_argument("--workspace_z_min", type=float, default=0.0, help="Workspace Z lower bound (Pass B)")
    parser.add_argument("--workspace_z_max", type=float, default=0.3, help="Workspace Z upper bound (Pass B)")
    
    args = parser.parse_args()
    
    train = not args.val
    
    if args.pass_type in ["a", "ab"]:
        mine_concepts_pass_a(
            dataset_name=args.dataset_name,
            data_dir=Path(args.data_dir),
            output_path=Path(args.output_path) if args.pass_type == "a" else Path(args.output_path) / "pass_a",
            train=train,
            gripper_closed_threshold=args.gripper_closed_threshold,
            arm_moving_epsilon=args.arm_moving_epsilon,
            table_height=args.table_height,
        )
    
    if args.pass_type in ["b", "ab"]:
        pass_a_path = Path(args.pass_a_concepts_path) if args.pass_a_concepts_path else (
            Path(args.output_path) / "pass_a" if args.pass_type == "ab" else None
        )
        mine_concepts_pass_b(
            dataset_name=args.dataset_name,
            data_dir=Path(args.data_dir),
            output_path=Path(args.output_path) if args.pass_type == "b" else Path(args.output_path) / "pass_b",
            train=train,
            confidence_threshold=args.confidence_threshold,
            pass_a_concepts_path=pass_a_path,
            max_trajectories=args.max_trajectories,
            save_images=args.save_images,
            use_depth=args.use_depth,
            depth_model=args.depth_model,
            workspace_x_min=args.workspace_x_min,
            workspace_x_max=args.workspace_x_max,
            workspace_y_min=args.workspace_y_min,
            workspace_y_max=args.workspace_y_max,
            workspace_z_min=args.workspace_z_min,
            workspace_z_max=args.workspace_z_max,
            gripper_closed_threshold=args.gripper_closed_threshold,
        )


if __name__ == "__main__":
    main()

