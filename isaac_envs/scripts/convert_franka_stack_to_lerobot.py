#!/usr/bin/env python3
"""
Convert Franka stack HDF5 to GR00T LeRobot v2 format.

Follows the schema in third_party/groot/getting_started/data_preparation.md:
- meta/: episodes.jsonl, modality.json, info.json, tasks.jsonl, stats.json
- data/chunk-*/: parquet per episode with observation.state, action, timestamp, ...
- videos/chunk-*/observation.images.<name>/: one MP4 per episode per camera (ego_view, wrist_view)

Videos are written only when the HDF5 contains obs/table_cam and obs/wrist_cam
(T, H, W, 3) uint8. 

Dependencies: h5py, numpy, pandas. For video writing: imageio, imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Project root = parent of isaac_envs
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FPS = 30
CHUNK_SIZE = 1000
TASK_DESCRIPTION = "stack the cubes"
# Franka stack env: table cam (ego) + wrist cam. LeRobot keys = observation.images.<name>
VIDEO_KEY_EGO = "observation.images.ego_view"    # table / external view
VIDEO_KEY_WRIST = "observation.images.wrist_view"  # wrist camera
VIDEO_KEYS = [VIDEO_KEY_EGO, VIDEO_KEY_WRIST]
# HDF5 obs keys for cameras (must be (T, H, W, 3) uint8)
HDF5_KEY_TABLE_CAM = "table_cam"
HDF5_KEY_WRIST_CAM = "wrist_cam"

# State: only end-effector pose (eef_pos + eef_quat). Total 7.
STATE_KEYS = ["eef_pos", "eef_quat"]
STATE_DIMS = [3, 4]
ACTION_DIM = 7


def build_modality_json(state_dims: list[int], state_keys: list[str]) -> dict:
    start = 0
    state_spec = {}
    for k, d in zip(state_keys, state_dims):
        state_spec[k] = {"start": start, "end": start + d}
        start += d
    return {
        "state": state_spec,
        "action": {
            "eef_delta": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "ego_view": {"original_key": VIDEO_KEY_EGO},
            "wrist_view": {"original_key": VIDEO_KEY_WRIST},
        },
        "annotation": {
            "human.action.task_description": {},
        },
    }


def _get_video_writer(path: Path):
    try:
        import imageio
    except ImportError:
        raise ImportError("Install imageio and imageio-ffmpeg for video writing: pip install imageio imageio-ffmpeg")
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=FPS, codec="libx264", format="FFMPEG")


def write_video_from_frames(path: Path, frames: np.ndarray) -> None:
    """Write (T, H, W, 3) uint8 array to MP4. Frames must be RGB.
    
    Pads frames to be divisible by 16 to avoid FFmpeg resizing warnings.
    Uses black padding (zeros) to maintain compatibility with video codecs.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape (T, H, W, 3), got {frames.shape}")
    
    T, H, W, C = frames.shape
    MACRO_BLOCK_SIZE = 16
    
    # Pad dimensions to be divisible by macro_block_size
    H_padded = ((H + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE) * MACRO_BLOCK_SIZE
    W_padded = ((W + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE) * MACRO_BLOCK_SIZE
    
    if H_padded != H or W_padded != W:
        # Pad frames with black (zeros) to avoid FFmpeg resizing
        padded_frames = np.zeros((T, H_padded, W_padded, C), dtype=frames.dtype)
        padded_frames[:, :H, :W, :] = frames
        frames = padded_frames
    
    writer = _get_video_writer(path)
    for t in range(frames.shape[0]):
        writer.append_data(np.ascontiguousarray(frames[t]))
    writer.close()


def run(hdf5_path: Path, output_dir: Path, eval_proportion: float = 0.1) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)
    data_dir = output_dir / "data"
    videos_dir = output_dir / "videos"

    modality = build_modality_json(STATE_DIMS, STATE_KEYS)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    # tasks.jsonl: task_index 0 = task description (GROOT uses annotation.human.action.task_description only)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n")

    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        demo_keys = sorted([k for k in data_group.keys() if k.startswith("demo_")])

    if not demo_keys:
        raise SystemExit(f"No demo_* keys found in {hdf5_path}. Nothing to convert.")

    all_state_arrays = []
    all_action_arrays = []
    episodes_meta = []
    global_index = 0
    total_videos = 0  # Track video count during conversion
    video_shapes = {}  # Track video dimensions: {video_key: (H, W, C)}

    for episode_index, demo_key in enumerate(demo_keys):
        with h5py.File(hdf5_path, "r") as f:
            d = f["data"][demo_key]
            T = d["actions"].shape[0]
            obs = d["obs"]
            # State: only eef_pos (3) + eef_quat (4)
            state_parts = [
                np.asarray(obs["eef_pos"], dtype=np.float32),
                np.asarray(obs["eef_quat"], dtype=np.float32),
            ]
            observation_state = np.concatenate(state_parts, axis=1)
            actions = np.asarray(d["actions"], dtype=np.float32)
            if actions.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"{demo_key}: actions shape {actions.shape} does not match ACTION_DIM={ACTION_DIM}"
                )
            # Load camera frames into memory so we can use them after the file is closed
            table_cam = np.asarray(obs[HDF5_KEY_TABLE_CAM], dtype=np.uint8) if HDF5_KEY_TABLE_CAM in obs else None
            wrist_cam = np.asarray(obs[HDF5_KEY_WRIST_CAM], dtype=np.uint8) if HDF5_KEY_WRIST_CAM in obs else None

        all_state_arrays.append(observation_state)
        all_action_arrays.append(actions)
        # tasks field should contain task text strings, not indices
        episodes_meta.append({"episode_index": episode_index, "tasks": [TASK_DESCRIPTION], "length": T})

        # Parquet per episode
        chunk_idx = episode_index // CHUNK_SIZE
        chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_dir / f"episode_{episode_index:06d}.parquet"

        timestamps = np.arange(T, dtype=np.float32) / FPS  # float32 to match example format
        episode_indices = np.full(T, episode_index, dtype=np.int64)
        indices = np.arange(global_index, global_index + T, dtype=np.int64)
        frame_indices = np.arange(T, dtype=np.int64)  # Frame index within episode (0 to T-1)
        global_index += T
        task_indices = np.zeros(T, dtype=np.int64)
        # GROOT parquet: annotation.human.action.task_description is index into meta/tasks.jsonl (0 = task text)
        annotation_task_description = np.zeros(T, dtype=np.int64)

        # Convert arrays to lists for parquet storage (each row = one timestep = one array)
        # observation_state and actions are (T, D) arrays, so list() creates list of T arrays
        df = pd.DataFrame({
            "observation.state": [row.tolist() for row in observation_state],  # Ensure Python lists, not numpy arrays
            "action": [row.tolist() for row in actions],  # Ensure Python lists, not numpy arrays
            "timestamp": timestamps,
            "frame_index": frame_indices,  # Frame index within episode (matches example)
            "annotation.human.action.task_description": annotation_task_description,
            "task_index": task_indices,
            "episode_index": episode_indices,
            "index": indices,  # Global frame index across all episodes
        })
        df.to_parquet(parquet_path, index=False)

        # Videos: only write when HDF5 has camera data (table_cam -> ego_view, wrist_cam -> wrist_view)
        if table_cam is not None:
            vid_dir = videos_dir / f"chunk-{chunk_idx:03d}" / VIDEO_KEY_EGO
            vid_dir.mkdir(parents=True, exist_ok=True)
            write_video_from_frames(vid_dir / f"episode_{episode_index:06d}.mp4", table_cam)
            total_videos += 1
            # Capture video dimensions (after padding for codec compatibility)
            if VIDEO_KEY_EGO not in video_shapes:
                T, H, W, C = table_cam.shape
                MACRO_BLOCK_SIZE = 16
                H_padded = ((H + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE) * MACRO_BLOCK_SIZE
                W_padded = ((W + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE) * MACRO_BLOCK_SIZE
                video_shapes[VIDEO_KEY_EGO] = (H_padded, W_padded, C)  # Actual video file dimensions
        if wrist_cam is not None:
            vid_dir = videos_dir / f"chunk-{chunk_idx:03d}" / VIDEO_KEY_WRIST
            vid_dir.mkdir(parents=True, exist_ok=True)
            write_video_from_frames(vid_dir / f"episode_{episode_index:06d}.mp4", wrist_cam)
            total_videos += 1
            # Capture video dimensions (after padding for codec compatibility)
            if VIDEO_KEY_WRIST not in video_shapes:
                T, H, W, C = wrist_cam.shape
                MACRO_BLOCK_SIZE = 16
                H_padded = ((H + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE) * MACRO_BLOCK_SIZE
                W_padded = ((W + MACRO_BLOCK_SIZE - 1) // MACRO_BLOCK_SIZE) * MACRO_BLOCK_SIZE
                video_shapes[VIDEO_KEY_WRIST] = (H_padded, W_padded, C)  # Actual video file dimensions

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for rec in episodes_meta:
            f.write(json.dumps(rec) + "\n")

    num_episodes = len(episodes_meta)
    all_states = np.vstack(all_state_arrays)
    all_actions = np.vstack(all_action_arrays)

    # ddof=0 avoids NaN std when there is only one frame; GROOT stats typically use population stats
    stats = {
        "observation.state": {
            "mean": np.mean(all_states, axis=0).tolist(),
            "std": np.std(all_states, axis=0, ddof=0).tolist(),
            "min": np.min(all_states, axis=0).tolist(),
            "max": np.max(all_states, axis=0).tolist(),
            "q01": np.quantile(all_states, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_states, 0.99, axis=0).tolist(),
        },
        "action": {
            "mean": np.mean(all_actions, axis=0).tolist(),
            "std": np.std(all_actions, axis=0, ddof=0).tolist(),
            "min": np.min(all_actions, axis=0).tolist(),
            "max": np.max(all_actions, axis=0).tolist(),
            "q01": np.quantile(all_actions, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_actions, 0.99, axis=0).tolist(),
        },
    }
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Path templates use episode_chunk and (for video) video_key per GROOT loader (lerobot_episode_loader.py).
    # Count total chunks (total_videos already counted during loop)
    total_chunks = (num_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
    
    # Calculate train/eval splits based on episode indices (last eval_proportion goes to eval, rest to train)
    # Format: episode index ranges like "0:9" means episodes 0 to 8 (inclusive start, exclusive end)
    train_end_idx = int((1.0 - eval_proportion) * num_episodes)
    splits = {
        "train": f"0:{train_end_idx}",
        "eval": f"{train_end_idx}:{num_episodes}",
    }
    
    # Build features dict with detailed metadata
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [sum(STATE_DIMS)],
            "names": STATE_KEYS,
        },
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": [
                "eef_delta_pos_x",  # Position delta (m)
                "eef_delta_pos_y",  # Position delta (m)
                "eef_delta_pos_z",  # Position delta (m)
                "eef_delta_rot_axis_x",  # Rotation delta: axis-angle x component (rad)
                "eef_delta_rot_axis_y",  # Rotation delta: axis-angle y component (rad)
                "eef_delta_rot_axis_z",  # Rotation delta: axis-angle z component (rad)
                "gripper",  # Gripper open/close (-1 open, +1 close)
            ],
        },
    }
    # Add video features with detailed metadata (matching demo_data format)
    for video_key in VIDEO_KEYS:
        if video_key in video_shapes:
            H, W, C = video_shapes[video_key]
            features[video_key] = {
                "dtype": "video",
                "shape": [H, W, C],
                "names": ["height", "width", "channels"],  # Note: "channels" plural, not "channel"
                "info": {  # Note: "info" not "video_info"
                    "video.height": int(H),
                    "video.width": int(W),
                    "video.codec": "h264",  # libx264 codec used by imageio
                    "video.pix_fmt": "yuv420p",  # Standard for H.264
                    "video.is_depth_map": False,
                    "video.fps": float(FPS),
                    "video.channels": int(C),
                    "has_audio": False,
                },
            }
        else:
            # Video not present in dataset, use minimal metadata
            features[video_key] = {"dtype": "video"}
    # Add standard columns (matching demo_data format exactly)
    features.update({
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},  # Frame index within episode
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},  # Global frame index
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        # Note: annotation is in parquet files but optional in features
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1], "names": None},
    })
    
    info = {
        "codebase_version": "v2.1",
        "total_episodes": num_episodes,
        "total_frames": int(all_states.shape[0]),
        "total_tasks": 1,  # Only one task: "stack the cubes"
        "fps": FPS,
        "chunks_size": CHUNK_SIZE,
        "splits": splits,  # Train/eval split based on episode percentages
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "total_chunks": total_chunks,
        "total_videos": total_videos,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote LeRobot dataset to {output_dir}")
    print(f"  Episodes: {num_episodes}, Frames: {all_states.shape[0]}")
    print(f"  Parquets: data/chunk-*/episode_*.parquet")
    print("  Videos: videos/chunk-*/{observation.images.ego_view,observation.images.wrist_view}/episode_*.mp4")


def main():
    parser = argparse.ArgumentParser(description="Convert Franka stack HDF5 to GROOT LeRobot v2 format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_PROJECT_ROOT / "isaac_envs" / "data" / "generated_dataset_small.hdf5",
        help="Input HDF5 path (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "isaac_envs" / "data" / "generated_dataset_small_lerobot",
        help="Output directory for LeRobot dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--eval-proportion",
        type=float,
        default=0.1,
        help="Proportion of episodes to use for evaluation (default: 0.1, i.e., last 10%%)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    if not 0.0 < args.eval_proportion < 1.0:
        parser.error(f"eval-proportion must be between 0 and 1, got {args.eval_proportion}")
    
    run(args.input, args.output, eval_proportion=args.eval_proportion)


if __name__ == "__main__":
    main()
