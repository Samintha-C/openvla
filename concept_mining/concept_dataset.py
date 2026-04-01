"""Merge multiple concept-mining passes into a single unified pseudo-label dataset.

Each pass is a HuggingFace Dataset saved to disk with the schema:
    {"dataset_name": str, "episode_id": str, "frame_index": int, "concepts": dict}

The merge keys on (episode_id, frame_index).  Concepts from later passes
overwrite earlier passes when keys collide (with a warning).

Usage:
    # Merge two passes
    python -m concept_mining.concept_dataset \
        --passes /path/to/pass_a /path/to/pass_b \
        --output /path/to/merged

    # Merge with explicit pass names (for logging)
    python -m concept_mining.concept_dataset \
        --passes /path/to/pass_a /path/to/pass_b \
        --pass_names a b \
        --output /path/to/merged

    # Require all passes to cover every frame (strict mode)
    python -m concept_mining.concept_dataset \
        --passes /path/to/pass_a /path/to/pass_b \
        --output /path/to/merged \
        --strict
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasets import Dataset, load_from_disk


FrameKey = Tuple[str, int]  # (episode_id, frame_index)


def load_pass(path: Path, pass_name: str) -> Dict[FrameKey, Dict]:
    """Load a pass dataset and return a dict keyed by (episode_id, frame_index).

    Each value is the full record dict (dataset_name, episode_id, frame_index,
    concepts).
    """
    ds = load_from_disk(str(path))
    print(f"  [{pass_name}] Loaded {len(ds)} records from {path}")

    records: Dict[FrameKey, Dict] = {}
    for r in ds:
        key = (r["episode_id"], r["frame_index"])
        records[key] = r
    return records


def merge_passes(
    pass_records: List[Dict[FrameKey, Dict]],
    pass_names: List[str],
    strict: bool = False,
) -> List[Dict]:
    """Merge N pass record dicts into a single list of unified records.

    Args:
        pass_records: List of dicts, one per pass, keyed by (episode_id, frame_index).
        pass_names: Human-readable name for each pass (for diagnostics).
        strict: If True, raise an error when a frame is missing from any pass.

    Returns:
        List of merged record dicts, sorted by (episode_id, frame_index).
    """
    # Collect the union of all frame keys
    all_keys: Set[FrameKey] = set()
    for records in pass_records:
        all_keys.update(records.keys())

    print(f"\nMerge summary:")
    print(f"  Total unique frames across all passes: {len(all_keys)}")

    # Per-pass coverage
    for name, records in zip(pass_names, pass_records):
        coverage = len(set(records.keys()) & all_keys)
        pct = 100.0 * coverage / len(all_keys) if all_keys else 0
        print(f"  [{name}] covers {coverage}/{len(all_keys)} frames ({pct:.1f}%)")

    # Detect concept key collisions across passes
    pass_concept_keys: List[Set[str]] = []
    for name, records in zip(pass_names, pass_records):
        concepts_seen: Set[str] = set()
        for r in records.values():
            concepts_seen.update(r["concepts"].keys())
        pass_concept_keys.append(concepts_seen)
        print(f"  [{name}] concept keys: {sorted(concepts_seen)}")

    for i in range(len(pass_names)):
        for j in range(i + 1, len(pass_names)):
            overlap = pass_concept_keys[i] & pass_concept_keys[j]
            if overlap:
                print(
                    f"  WARNING: concept key overlap between [{pass_names[i]}] "
                    f"and [{pass_names[j]}]: {sorted(overlap)}. "
                    f"Later pass ({pass_names[j]}) will overwrite."
                )

    # Merge
    missing_report: Dict[str, int] = {name: 0 for name in pass_names}
    merged: Dict[FrameKey, Dict] = {}

    for key in all_keys:
        # Start from the first pass that has this key (for dataset_name, episode_id, etc.)
        base_record = None
        merged_concepts: Dict[str, float] = {}

        for name, records in zip(pass_names, pass_records):
            if key in records:
                r = records[key]
                if base_record is None:
                    base_record = {
                        "dataset_name": r["dataset_name"],
                        "episode_id": r["episode_id"],
                        "frame_index": r["frame_index"],
                    }
                merged_concepts.update(r["concepts"])
            else:
                missing_report[name] += 1

        base_record["concepts"] = merged_concepts
        merged[key] = base_record

    # Report missing frames
    for name, count in missing_report.items():
        if count > 0:
            pct = 100.0 * count / len(all_keys)
            msg = (
                f"  [{name}] missing from {count}/{len(all_keys)} "
                f"frames ({pct:.1f}%)"
            )
            if strict:
                raise ValueError(
                    f"Strict mode: {msg}. All passes must cover every frame."
                )
            print(f"  WARNING: {msg} — those frames will lack [{name}] concepts.")

    # Sort by episode_id then frame_index for deterministic output
    sorted_records = sorted(merged.values(), key=lambda r: (r["episode_id"], r["frame_index"]))

    all_concept_keys = set()
    for r in sorted_records:
        all_concept_keys.update(r["concepts"].keys())
    print(f"\n  Merged concept keys ({len(all_concept_keys)}): {sorted(all_concept_keys)}")
    print(f"  Total merged records: {len(sorted_records)}")

    return sorted_records


def build_dataset(
    pass_paths: List[Path],
    pass_names: Optional[List[str]] = None,
    strict: bool = False,
) -> Dataset:
    """Load passes from disk, merge, and return a HuggingFace Dataset.

    Args:
        pass_paths: Paths to saved HuggingFace Datasets, one per pass.
        pass_names: Optional human-readable names. Defaults to pass_a, pass_b, ...
        strict: If True, raise on incomplete coverage.

    Returns:
        Merged HuggingFace Dataset.
    """
    if pass_names is None:
        pass_names = [chr(ord("a") + i) for i in range(len(pass_paths))]
    if len(pass_names) != len(pass_paths):
        raise ValueError(
            f"pass_names length ({len(pass_names)}) must match "
            f"pass_paths length ({len(pass_paths)})"
        )

    print(f"Loading {len(pass_paths)} passes...")
    pass_records = []
    for path, name in zip(pass_paths, pass_names):
        records = load_pass(path, name)
        pass_records.append(records)

    merged_records = merge_passes(pass_records, pass_names, strict=strict)
    return Dataset.from_list(merged_records)


def main():
    parser = argparse.ArgumentParser(
        description="Merge concept-mining passes into a unified pseudo-label dataset"
    )
    parser.add_argument(
        "--passes",
        type=str,
        nargs="+",
        required=True,
        help="Paths to saved concept datasets, one per pass (order matters for overwrites)",
    )
    parser.add_argument(
        "--pass_names",
        type=str,
        nargs="+",
        default=None,
        help="Human-readable names for each pass (default: a, b, c, ...)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the merged concept dataset",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any pass is missing frames that other passes have",
    )
    args = parser.parse_args()

    pass_paths = [Path(p) for p in args.passes]
    for p in pass_paths:
        if not p.exists():
            raise FileNotFoundError(f"Pass dataset not found: {p}")

    output_path = Path(args.output)

    dataset = build_dataset(pass_paths, args.pass_names, strict=args.strict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    print(f"\nSaved merged dataset to {output_path}")

    if len(dataset) > 0:
        print(f"Sample record: {dataset[0]}")


if __name__ == "__main__":
    main()
