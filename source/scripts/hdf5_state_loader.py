#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Load HDF5 demo state into Isaac Lab scene.get_state() format for reset_to().

The loader is scene-agnostic: you pass a SceneStateConfig that lists which
articulation and rigid-object entities to load 
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import torch


# -----------------------------------------------------------------------------
# Scene config: which entities to load and how they map to HDF5
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneStateConfig:
    """Which scene entities to load and optional scene_name -> HDF5 group name mapping.

    HDF5 layout is fixed: data/demo_<n>/initial_state|states/articulation|rigid_object/<name>/.
    """

    articulation_names: tuple[str, ...] = ()
    rigid_object_names: tuple[str, ...] = ()
    articulation_hdf5_names: dict[str, str] = field(default_factory=dict)
    rigid_object_hdf5_names: dict[str, str] = field(default_factory=dict)

    def get_articulation_hdf5_name(self, scene_name: str) -> str:
        return self.articulation_hdf5_names.get(scene_name, scene_name)

    def get_rigid_object_hdf5_name(self, scene_name: str) -> str:
        return self.rigid_object_hdf5_names.get(scene_name, scene_name)


# Franka stack cube scene: one robot, three cubes (matches scene registration).
FRANKA_STACK_SCENE_CONFIG = SceneStateConfig(
    articulation_names=("robot",),
    rigid_object_names=("cube_1", "cube_2", "cube_3"),
)


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------

def load_demo_state_from_hdf5(
    hdf5_path: Path | str,
    demo_index: int,
    step_index: int,
    device: torch.device,
    num_envs: int = 2,
    scene_config: SceneStateConfig = FRANKA_STACK_SCENE_CONFIG,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Load state at (demo_index, step_index) from HDF5 and return scene state for reset_to().

    If step_index is 0, reads from initial_state/ (shape (1,...)); otherwise reads from
    states/ at that step. Result tensors are broadcast to (num_envs, ...) so the same
    state can be applied to multiple envs.

    Args:
        hdf5_path: Path to the HDF5 file (Franka stack or other scene matching scene_config).
        demo_index: Demo index (e.g. 0 for demo_0).
        step_index: Step within the demo (0 = initial_state; else index into states/).
        device: Device for output tensors.
        num_envs: Batch size for each tensor (e.g. 2 for comparison mode).
        scene_config: Which entities to load and HDF5 name mapping. 

    Returns:
        Nested dict compatible with InteractiveScene.reset_to():
        {"articulation": {<name>: {...}}, "rigid_object": {<name>: {...}}}
    """
    hdf5_path = Path(hdf5_path)
    demo_key = f"demo_{demo_index}"
    state: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        "articulation": {},
        "rigid_object": {},
    }

    def to_tensor(arr: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        t = torch.from_numpy(arr).to(dtype=dtype, device=device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and num_envs > 1:
            t = t.repeat(num_envs, *([1] * (t.dim() - 1)))
        return t

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError(f"Group 'data' not found in {hdf5_path}")
        data = f["data"]
        if demo_key not in data:
            raise KeyError(f"Demo {demo_key} not found in {hdf5_path}")

        d = data[demo_key]
        if "actions" not in d:
            raise KeyError(f"Demo {demo_key} must contain 'actions' to infer length")
        T = d["actions"].shape[0]
        if step_index < 0 or step_index >= T:
            raise ValueError(f"step_index {step_index} out of range [0, {T}) for {demo_key}")

        use_initial = step_index == 0
        base = d["initial_state"] if use_initial else d["states"]
        slice_spec = slice(0, 1) if use_initial else slice(step_index, step_index + 1)

        # Articulations
        if "articulation" in base:
            for scene_name in scene_config.articulation_names:
                hdf5_name = scene_config.get_articulation_hdf5_name(scene_name)
                if hdf5_name not in base["articulation"]:
                    continue
                art = base["articulation"][hdf5_name]
                state["articulation"][scene_name] = {
                    "joint_position": to_tensor(art["joint_position"][slice_spec]),
                    "joint_velocity": to_tensor(art["joint_velocity"][slice_spec]),
                    "root_pose": to_tensor(art["root_pose"][slice_spec]),
                    "root_velocity": to_tensor(art["root_velocity"][slice_spec]),
                }

        # Rigid objects
        if "rigid_object" in base:
            for scene_name in scene_config.rigid_object_names:
                hdf5_name = scene_config.get_rigid_object_hdf5_name(scene_name)
                if hdf5_name not in base["rigid_object"]:
                    continue
                robj = base["rigid_object"][hdf5_name]
                state["rigid_object"][scene_name] = {
                    "root_pose": to_tensor(robj["root_pose"][slice_spec]),
                    "root_velocity": to_tensor(robj["root_velocity"][slice_spec]),
                }

    return state
