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

## Analyzing Mined Concepts

After mining concepts, you can analyze them:

```bash
python -m concept_mining.analyze_concepts \
  --concept_dataset_path /path/to/concepts/bridge_orig_pass_a \
  --output_path ./concept_analysis.json
```

This will print statistics about:
- Concept distributions (mean, std, positive rates)
- Episode-level statistics
- Sample records

## Visualizing Concepts

For detailed visualizations and analysis:

```bash
python -m concept_mining.visualize_concepts \
  --concept_dataset_path /path/to/concepts/bridge_orig_pass_a \
  --output_dir ./concept_visualizations \
  --num_examples 20
```

This will:
- Show concept statistics and distributions
- Create plots of concept distributions
- Show concept co-occurrence matrix
- Display examples of frames with different concept combinations
- Analyze episode-level patterns (how concepts change within episodes)
- Save all plots and statistics to the output directory

## Viewing Concept Images (Sanity Check)

To extract and view example images that correspond to concept records (run in pod):

```bash
# In the pod or via kubectl exec
python -m concept_mining.view_concept_images \
  --concept_dataset_path /sc-cbint-vol/openvla-concepts/bridge_orig_pass_a \
  --rlds_data_dir /sc-cbint-vol/rlds_data \
  --dataset_name bridge_orig \
  --output_dir /sc-cbint-vol/concept_examples \
  --num_examples_per_concept 5
```

This will:
- Load the concept dataset
- Find examples of frames with each concept active
- Extract corresponding images from the RLDS dataset
- Save example images to the output directory

**To run from local (via kubectl exec):**
```bash
kubectl exec -it sc-copy-pod -n wenglab-interpretable-ai -- sh -c "cd /workspace/openvla && python -m concept_mining.view_concept_images --concept_dataset_path /sc-cbint-vol/openvla-concepts/bridge_orig_pass_a --rlds_data_dir /sc-cbint-vol/rlds_data --dataset_name bridge_orig --output_dir /sc-cbint-vol/concept_examples --num_examples_per_concept 5"
```

**Or if the pod doesn't have /workspace/openvla, use the repo path directly:**
```bash
kubectl exec -it sc-copy-pod -n wenglab-interpretable-ai -- sh -c "python -m concept_mining.view_concept_images --concept_dataset_path /sc-cbint-vol/openvla-concepts/bridge_orig_pass_a --rlds_data_dir /sc-cbint-vol/rlds_data --dataset_name bridge_orig --output_dir /sc-cbint-vol/concept_examples --num_examples_per_concept 5"
```

Then copy the images to local:
```bash
kubectl exec sc-copy-pod -n wenglab-interpretable-ai -- tar czf /tmp/concept_examples.tar.gz -C /sc-cbint-vol concept_examples && \
kubectl cp wenglab-interpretable-ai/sc-copy-pod:/tmp/concept_examples.tar.gz ./concept_examples.tar.gz && \
tar xzf ./concept_examples.tar.gz && \
rm ./concept_examples.tar.gz && \
kubectl exec sc-copy-pod -n wenglab-interpretable-ai -- rm /tmp/concept_examples.tar.gz
```

## Copying Concepts to Local

To copy concepts from pod to local:

```bash
# Using tar (recommended for directories)
kubectl exec -it sc-cbvlam-pod -n wenglab-interpretable-ai -- tar czf - -C /sc-cbint-vol/openvla-concepts bridge_orig_pass_a | tar xzf - -C ./local/concepts/
```

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

