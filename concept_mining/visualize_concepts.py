import argparse
from pathlib import Path
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")
from datasets import load_from_disk
import json
from collections import defaultdict


def load_concepts(concept_dataset_path: Path):
    """Load concept dataset."""
    print(f"Loading concept dataset from {concept_dataset_path}...")
    dataset = load_from_disk(str(concept_dataset_path))
    print(f"Loaded {len(dataset)} concept records")
    return dataset


def analyze_concept_statistics(dataset):
    """Compute and display concept statistics."""
    if len(dataset) == 0:
        print("Empty dataset!")
        return None
    
    concept_names = list(dataset[0]["concepts"].keys())
    print(f"\nConcepts found: {concept_names}")
    
    all_concepts = {name: [] for name in concept_names}
    episode_ids = set()
    
    for record in dataset:
        episode_ids.add(record["episode_id"])
        for concept_name in concept_names:
            all_concepts[concept_name].append(record["concepts"][concept_name])
    
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
            "total_positive": int(np.sum(values)),
            "total_frames": len(values),
        }
    
    print(f"\n{'='*70}")
    print("CONCEPT STATISTICS")
    print(f"{'='*70}")
    print(f"Total episodes: {len(episode_ids)}")
    print(f"Total frames: {len(dataset)}")
    print(f"Average frames per episode: {len(dataset) / len(episode_ids):.1f}\n")
    
    for concept_name in concept_names:
        s = stats[concept_name]
        print(f"{concept_name}:")
        print(f"  Positive rate: {s['positive_rate']:.2%} ({s['total_positive']}/{s['total_frames']})")
        print(f"  Mean value: {s['mean']:.4f}")
        print(f"  Std dev: {s['std']:.4f}")
        print(f"  Range: [{s['min']:.4f}, {s['max']:.4f}]")
        print()
    
    return stats, all_concepts, concept_names


def plot_concept_distributions(all_concepts, concept_names, output_dir: Path):
    """Create visualization plots for concept distributions."""
    if not HAS_PLOTTING:
        print("Skipping plots (matplotlib/seaborn not available)")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_concepts = len(concept_names)
    fig, axes = plt.subplots(1, n_concepts, figsize=(5*n_concepts, 4))
    if n_concepts == 1:
        axes = [axes]
    
    for idx, concept_name in enumerate(concept_names):
        values = np.array(all_concepts[concept_name])
        ax = axes[idx]
        
        unique, counts = np.unique(values, return_counts=True)
        ax.bar(unique, counts, alpha=0.7, color='steelblue')
        ax.set_xlabel('Concept Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{concept_name}\nPositive: {np.sum(values > 0)}/{len(values)} ({np.mean(values > 0):.1%})')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'concept_distributions.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {plot_path}")
    plt.close()


def plot_concept_cooccurrence(dataset, concept_names, output_dir: Path):
    """Plot concept co-occurrence matrix."""
    if not HAS_PLOTTING:
        print("Skipping co-occurrence plot (matplotlib/seaborn not available)")
        return
    
    cooccurrence = np.zeros((len(concept_names), len(concept_names)), dtype=np.int64)
    
    for record in dataset:
        concepts = record["concepts"]
        values = [concepts[name] for name in concept_names]
        for i in range(len(concept_names)):
            for j in range(len(concept_names)):
                if values[i] > 0 and values[j] > 0:
                    cooccurrence[i, j] += 1
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cooccurrence, annot=True, fmt='d', xticklabels=concept_names, 
                yticklabels=concept_names, cmap='YlOrRd', cbar_kws={'label': 'Co-occurrences'})
    plt.title('Concept Co-occurrence Matrix')
    plt.tight_layout()
    plot_path = output_dir / 'concept_cooccurrence.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved co-occurrence plot to {plot_path}")
    plt.close()


def show_concept_examples(dataset, concept_names, num_examples=10):
    """Show examples of frames with different concept combinations."""
    print(f"\n{'='*70}")
    print(f"CONCEPT EXAMPLES (showing {num_examples} examples)")
    print(f"{'='*70}\n")
    
    examples_by_concept = defaultdict(list)
    
    for record in dataset:
        concepts = record["concepts"]
        for concept_name in concept_names:
            if concepts[concept_name] > 0:
                examples_by_concept[concept_name].append(record)
    
    for concept_name in concept_names:
        examples = examples_by_concept[concept_name][:num_examples]
        print(f"\n{concept_name} = 1.0 (showing {len(examples)} examples):")
        for i, ex in enumerate(examples[:5]):
            print(f"  Example {i+1}:")
            print(f"    Episode: {ex['episode_id']}")
            print(f"    Frame: {ex['frame_index']}")
            print(f"    All concepts: {ex['concepts']}")
    
    print(f"\n\nRandom sample records:")
    for i in range(min(num_examples, len(dataset))):
        idx = np.random.randint(0, len(dataset))
        record = dataset[idx]
        print(f"\n  Record {i+1} (index {idx}):")
        print(f"    Episode: {record['episode_id']}")
        print(f"    Frame: {record['frame_index']}")
        print(f"    Concepts: {record['concepts']}")


def analyze_episode_patterns(dataset, concept_names):
    """Analyze how concepts change within episodes."""
    print(f"\n{'='*70}")
    print("EPISODE-LEVEL PATTERNS")
    print(f"{'='*70}\n")
    
    episodes = defaultdict(list)
    for record in dataset:
        episodes[record["episode_id"]].append(record)
    
    episode_stats = []
    for episode_id, records in list(episodes.items())[:10]:
        records = sorted(records, key=lambda x: x["frame_index"])
        episode_concepts = {name: [] for name in concept_names}
        for record in records:
            for concept_name in concept_names:
                episode_concepts[concept_name].append(record["concepts"][concept_name])
        
        stats = {
            "episode_id": episode_id,
            "num_frames": len(records),
            "concepts": {}
        }
        
        for concept_name in concept_names:
            values = np.array(episode_concepts[concept_name])
            stats["concepts"][concept_name] = {
                "positive_frames": int(np.sum(values > 0)),
                "positive_rate": float(np.mean(values > 0)),
                "transitions": int(np.sum(np.diff(values) != 0)),
            }
        
        episode_stats.append(stats)
        
        print(f"{episode_id}:")
        print(f"  Frames: {stats['num_frames']}")
        for concept_name in concept_names:
            c = stats["concepts"][concept_name]
            print(f"  {concept_name}: {c['positive_frames']}/{stats['num_frames']} frames ({c['positive_rate']:.1%}), {c['transitions']} transitions")
    
    if len(episodes) > 10:
        print(f"\n... (showing first 10 of {len(episodes)} episodes)")


def visualize_concepts(concept_dataset_path: Path, output_dir: Path = None, num_examples: int = 10):
    """
    Main visualization function.
    
    Args:
        concept_dataset_path: Path to concept dataset
        output_dir: Directory to save plots (optional)
        num_examples: Number of examples to show
    """
    dataset = load_concepts(concept_dataset_path)
    
    stats, all_concepts, concept_names = analyze_concept_statistics(dataset)
    if stats is None:
        return
    
    show_concept_examples(dataset, concept_names, num_examples)
    analyze_episode_patterns(dataset, concept_names)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_concept_distributions(all_concepts, concept_names, output_dir)
        plot_concept_cooccurrence(dataset, concept_names, output_dir)
        
        stats_output = {
            "dataset_path": str(concept_dataset_path),
            "total_records": len(dataset),
            "concept_names": concept_names,
            "statistics": stats,
        }
        
        stats_path = output_dir / "concept_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats_output, f, indent=2)
        print(f"\nSaved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize and analyze mined concepts")
    parser.add_argument("--concept_dataset_path", type=str, required=True, 
                       help="Path to concept dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save plots and statistics")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of examples to display")
    
    args = parser.parse_args()
    
    visualize_concepts(
        concept_dataset_path=Path(args.concept_dataset_path),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        num_examples=args.num_examples,
    )


if __name__ == "__main__":
    main()

