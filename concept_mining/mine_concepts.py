import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import tensorflow as tf
import dlimp as dl
from datasets import Dataset
from tqdm import tqdm

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, StateEncoding
from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
from concept_mining.extractors import ProprioceptiveExtractor


def get_state_encoding_str(state_encoding: StateEncoding) -> str:
    mapping = {
        StateEncoding.POS_EULER: "POS_EULER",
        StateEncoding.POS_QUAT: "POS_QUAT",
        StateEncoding.JOINT: "JOINT",
        StateEncoding.NONE: "NONE",
    }
    return mapping.get(state_encoding, "NONE")


def extract_concepts_from_trajectory(
    traj: Dict[str, Any],
    dataset_name: str,
    episode_idx: int,
    extractor: ProprioceptiveExtractor,
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
    
    extractor.reset()
    
    proprio = traj["observation"]["proprio"]
    
    if isinstance(proprio, tf.Tensor):
        traj_len = int(proprio.shape[0])
        proprio_np = proprio.numpy()
    else:
        traj_len = len(proprio)
        proprio_np = np.array(proprio)
    
    episode_id = f"{dataset_name}_episode_{episode_idx:06d}"
    
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
    
    dataset_config = OXE_DATASET_CONFIGS.get(dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
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
        dataset_name=dataset_name,
        data_root_dir=data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=True,
        load_language=False,
    )
    
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
    
    print(f"Iterating through trajectories...")
    for traj_idx, traj in enumerate(tqdm(ds.as_numpy_iterator())):
        if "observation" not in traj:
            continue
        
        if "proprio" not in traj["observation"]:
            continue
        
        concept_records = extract_concepts_from_trajectory(traj, dataset_name, traj_idx, extractor)
        all_concept_records.extend(concept_records)
        
        if (traj_idx + 1) % 100 == 0:
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


def main():
    parser = argparse.ArgumentParser(description="Mine Pass A (proprioceptive) concepts from RLDS datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="RLDS dataset name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing RLDS data")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save concept dataset")
    parser.add_argument("--train", action="store_true", default=True, help="Use train split (default: True)")
    parser.add_argument("--val", action="store_true", help="Use validation split")
    parser.add_argument("--gripper_closed_threshold", type=float, default=0.05, help="Gripper closed threshold")
    parser.add_argument("--arm_moving_epsilon", type=float, default=0.01, help="Arm moving epsilon threshold")
    parser.add_argument("--table_height", type=float, default=0.0, help="Table height threshold")
    
    args = parser.parse_args()
    
    train = not args.val
    
    mine_concepts_pass_a(
        dataset_name=args.dataset_name,
        data_dir=Path(args.data_dir),
        output_path=Path(args.output_path),
        train=train,
        gripper_closed_threshold=args.gripper_closed_threshold,
        arm_moving_epsilon=args.arm_moving_epsilon,
        table_height=args.table_height,
    )


if __name__ == "__main__":
    main()

