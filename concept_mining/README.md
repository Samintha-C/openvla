# Concept Mining Pipeline

This directory contains the offline concept mining pipeline for CB-OpenVLA.

## Pass A: Proprioceptive Heuristics

Extracts simple heuristics from robot proprioceptive state:
- `gripper_closed`: Binary indicator if gripper width < threshold
- `arm_moving`: Binary indicator if arm position change > epsilon
- `height_above_table`: Binary indicator if end-effector z > table_height

## Usage

```bash
python -m concept_mining.mine_concepts \
    --dataset_name bridge_oxe \
    --data_dir /path/to/rlds/data \
    --output_path /path/to/output/concepts \
    --train \
    --gripper_closed_threshold 0.05 \
    --arm_moving_epsilon 0.01 \
    --table_height 0.0
```

## Output Format

The mined concepts are saved as a HuggingFace Dataset with the following structure:

```python
{
    "dataset_name": str,
    "episode_id": str,  # Format: "{dataset_name}_episode_{idx:06d}"
    "frame_index": int,
    "concepts": {
        "gripper_closed": float,  # 0.0 or 1.0
        "arm_moving": float,      # 0.0 or 1.0
        "height_above_table": float  # 0.0 or 1.0
    }
}
```

The dataset is indexed by `(episode_id, frame_index)` for O(1) lookup during training.

