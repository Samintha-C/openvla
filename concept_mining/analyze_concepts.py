import argparse
from pathlib import Path
import numpy as np
from datasets import load_from_disk
import json


def analyze_concepts(concept_dataset_path: Path, output_path: Path = None):
    """
    Analyze mined concepts and generate statistics.
    
    Args:
        concept_dataset_path: Path to the concept dataset
        output_path: Optional path to save analysis JSON
    """
    print(f"Loading concept dataset from {concept_dataset_path}...")
    dataset = load_from_disk(str(concept_dataset_path))
    
    print(f"Dataset size: {len(dataset)} records")
    
    if len(dataset) == 0:
        print("Empty dataset!")
        return
    
    concept_names = list(dataset[0]["concepts"].keys())
    print(f"Concepts: {concept_names}")
    
    all_concepts = {name: [] for name in concept_names}
    episode_ids = set()
    
    for record in dataset:
        episode_ids.add(record["episode_id"])
        for concept_name in concept_names:
            all_concepts[concept_name].append(record["concepts"][concept_name])
    
    print(f"\nNumber of unique episodes: {len(episode_ids)}")
    print(f"Average frames per episode: {len(dataset) / len(episode_ids):.1f}")
    
    print("\n" + "="*60)
    print("CONCEPT STATISTICS")
    print("="*60)
    
    stats = {}
    for concept_name in concept_names:
        values = np.array(all_concepts[concept_name])
        stats[concept_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sum": int(np.sum(values)),
            "positive_rate": float(np.mean(values > 0)),
            "zero_rate": float(np.mean(values == 0)),
        }
        
        print(f"\n{concept_name}:")
        print(f"  Mean: {stats[concept_name]['mean']:.4f}")
        print(f"  Std:  {stats[concept_name]['std']:.4f}")
        print(f"  Range: [{stats[concept_name]['min']:.4f}, {stats[concept_name]['max']:.4f}]")
        print(f"  Positive rate: {stats[concept_name]['positive_rate']:.2%}")
        print(f"  Zero rate: {stats[concept_name]['zero_rate']:.2%}")
        print(f"  Total positive: {stats[concept_name]['sum']} / {len(values)}")
    
    print("\n" + "="*60)
    print("EPISODE-LEVEL STATISTICS")
    print("="*60)
    
    episode_stats = {}
    for episode_id in sorted(list(episode_ids))[:10]:
        episode_records = [r for r in dataset if r["episode_id"] == episode_id]
        episode_concepts = {name: [] for name in concept_names}
        for record in episode_records:
            for concept_name in concept_names:
                episode_concepts[concept_name].append(record["concepts"][concept_name])
        
        episode_stats[episode_id] = {
            "num_frames": len(episode_records),
            "concepts": {
                name: {
                    "mean": float(np.mean(episode_concepts[name])),
                    "sum": int(np.sum(episode_concepts[name])),
                }
                for name in concept_names
            }
        }
        
        print(f"\n{episode_id}:")
        print(f"  Frames: {episode_stats[episode_id]['num_frames']}")
        for concept_name in concept_names:
            mean_val = episode_stats[episode_id]["concepts"][concept_name]["mean"]
            sum_val = episode_stats[episode_id]["concepts"][concept_name]["sum"]
            print(f"  {concept_name}: mean={mean_val:.3f}, total={sum_val}")
    
    if len(episode_ids) > 10:
        print(f"\n... (showing first 10 of {len(episode_ids)} episodes)")
    
    if output_path:
        output_data = {
            "dataset_path": str(concept_dataset_path),
            "total_records": len(dataset),
            "num_episodes": len(episode_ids),
            "concept_names": concept_names,
            "concept_statistics": stats,
            "sample_episodes": episode_stats,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nAnalysis saved to {output_path}")
    
    print("\n" + "="*60)
    print("SAMPLE RECORDS")
    print("="*60)
    
    for i in [0, len(dataset)//2, len(dataset)-1]:
        record = dataset[i]
        print(f"\nRecord {i}:")
        print(f"  Episode: {record['episode_id']}")
        print(f"  Frame: {record['frame_index']}")
        print(f"  Concepts: {record['concepts']}")


def main():
    parser = argparse.ArgumentParser(description="Analyze mined concepts")
    parser.add_argument("--concept_dataset_path", type=str, required=True, help="Path to concept dataset")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save analysis JSON")
    
    args = parser.parse_args()
    
    analyze_concepts(
        concept_dataset_path=Path(args.concept_dataset_path),
        output_path=Path(args.output_path) if args.output_path else None,
    )


if __name__ == "__main__":
    main()

