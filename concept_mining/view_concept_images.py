import argparse
from pathlib import Path
import numpy as np
from datasets import load_from_disk
import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prismatic.vla.datasets.rlds.oxe.configs import OXE_DATASET_CONFIGS, StateEncoding
from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
import matplotlib.pyplot as plt
from PIL import Image


TFDS_TO_OXE_NAME_MAP = {
    "bridge": "bridge_oxe",
    "bridge_dataset": "bridge_oxe",
    "bridge_orig": "bridge_oxe",
}


def find_episode_in_dataset(dataset, episode_id):
    """Find a specific episode in the dataset."""
    episode_idx = None
    if "_episode_" in episode_id:
        try:
            idx_str = episode_id.split("_episode_")[1]
            episode_idx = int(idx_str)
        except:
            pass
    
    if episode_idx is not None:
        for idx, traj in enumerate(dataset.as_numpy_iterator()):
            if idx == episode_idx:
                return traj, idx
    
    return None, None


def extract_image_from_trajectory(traj, frame_index, camera_view="primary"):
    """Extract image from trajectory at specific frame index."""
    if "observation" not in traj:
        return None
    
    obs = traj["observation"]
    
    if "image" in obs:
        images = obs["image"]
        if isinstance(images, dict):
            if camera_view in images:
                img = images[camera_view]
            else:
                img = list(images.values())[0]
        else:
            img = images
    elif "rgb" in obs:
        img = obs["rgb"]
    else:
        return None
    
    if isinstance(img, tf.Tensor):
        img_np = img.numpy()
    else:
        img_np = np.array(img)
    
    if len(img_np.shape) == 4:
        img_np = img_np[frame_index]
    elif len(img_np.shape) == 3:
        img_np = img_np
    else:
        return None
    
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    return img_np


def view_concept_images(
    concept_dataset_path: Path,
    rlds_data_dir: Path,
    dataset_name: str,
    output_dir: Path = None,
    num_examples_per_concept: int = 5,
):
    """
    Extract and save example images for concept records.
    
    Args:
        concept_dataset_path: Path to concept dataset
        rlds_data_dir: Path to RLDS data directory
        dataset_name: Name of the RLDS dataset
        output_dir: Directory to save images
        num_examples_per_concept: Number of examples to extract per concept
    """
    print(f"Loading concept dataset from {concept_dataset_path}...")
    concept_dataset = load_from_disk(str(concept_dataset_path))
    print(f"Loaded {len(concept_dataset)} concept records")
    
    if len(concept_dataset) == 0:
        print("Empty concept dataset!")
        return
    
    concept_names = list(concept_dataset[0]["concepts"].keys())
    print(f"Concepts: {concept_names}")
    
    oxe_dataset_name = TFDS_TO_OXE_NAME_MAP.get(dataset_name, dataset_name)
    dataset_config = OXE_DATASET_CONFIGS.get(oxe_dataset_name)
    if dataset_config is None:
        print(f"Warning: Dataset {dataset_name} not found in OXE configs. Trying anyway...")
    
    dataset_kwargs = make_oxe_dataset_kwargs(
        dataset_name=oxe_dataset_name,
        data_root_dir=rlds_data_dir,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=True,
        load_language=True,
    )
    dataset_kwargs["name"] = dataset_name
    
    print(f"Loading RLDS dataset {dataset_name}...")
    try:
        ds, _ = make_dataset_from_rlds(
            train=True,
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
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    examples_by_concept = {name: [] for name in concept_names}
    
    for record in concept_dataset:
        concepts = record["concepts"]
        for concept_name in concept_names:
            if concepts[concept_name] > 0 and len(examples_by_concept[concept_name]) < num_examples_per_concept:
                examples_by_concept[concept_name].append(record)
    
    print(f"\n{'='*70}")
    print("EXTRACTING EXAMPLE IMAGES")
    print(f"{'='*70}\n")
    
    episode_cache = {}
    
    for concept_name in concept_names:
        examples = examples_by_concept[concept_name]
        print(f"\n{concept_name} ({len(examples)} examples):")
        
        for ex_idx, example in enumerate(examples):
            episode_id = example["episode_id"]
            frame_index = example["frame_index"]
            
            if episode_id not in episode_cache:
                episode_idx = None
                if "_episode_" in episode_id:
                    try:
                        idx_str = episode_id.split("_episode_")[1]
                        episode_idx = int(idx_str)
                    except:
                        pass
                
                if episode_idx is not None:
                    for idx, traj in enumerate(ds.as_numpy_iterator()):
                        if idx == episode_idx:
                            episode_cache[episode_id] = traj
                            break
                else:
                    print(f"  Example {ex_idx+1}: Could not parse episode_id {episode_id}")
                    continue
            else:
                traj = episode_cache[episode_id]
            
            img = extract_image_from_trajectory(traj, frame_index)
            
            if img is not None:
                print(f"  Example {ex_idx+1}: Episode {episode_id}, Frame {frame_index} - Image shape: {img.shape}")
                
                if output_dir:
                    img_path = output_dir / f"{concept_name}_example_{ex_idx+1}_ep{episode_id.split('_')[-1]}_frame{frame_index}.png"
                    Image.fromarray(img).save(img_path)
                    print(f"    Saved to {img_path}")
            else:
                print(f"  Example {ex_idx+1}: Could not extract image from Episode {episode_id}, Frame {frame_index}")
    
    if output_dir:
        print(f"\nAll images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="View example images for concept records")
    parser.add_argument("--concept_dataset_path", type=str, required=True,
                       help="Path to concept dataset")
    parser.add_argument("--rlds_data_dir", type=str, required=True,
                       help="Path to RLDS data directory")
    parser.add_argument("--dataset_name", type=str, required=True,
                       help="RLDS dataset name (e.g., bridge_orig)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save example images")
    parser.add_argument("--num_examples_per_concept", type=int, default=5,
                       help="Number of examples to extract per concept")
    
    args = parser.parse_args()
    
    view_concept_images(
        concept_dataset_path=Path(args.concept_dataset_path),
        rlds_data_dir=Path(args.rlds_data_dir),
        dataset_name=args.dataset_name,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        num_examples_per_concept=args.num_examples_per_concept,
    )


if __name__ == "__main__":
    main()

