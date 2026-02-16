# Franka stack cube HDF5 dataset format

This document describes the structure of the Franka stack cube demonstration dataset stored in HDF5 (e.g. `data/franka_stack_dataset.hdf5`). Use `scripts/print_hdf5_structure.py` to inspect any such file.

## Overview

- **Root**: Single top-level group `data/`.
- **Demos**: Each demonstration is a group under `data/` with a key like `demo_0`, `demo_1`, etc. Demo keys may be non-sequential (e.g. `demo_0`, `demo_1`, `demo_2`, `demo_6`, …).
- **Trajectory length**: Number of timesteps varies per demo (e.g. 179–247 steps). All arrays in that demo use the same length `T` for the time dimension.
- **Dtypes**: All arrays are `float32` unless noted otherwise.

## Per-demo layout

Each `data/demo_<n>/` group contains:

| Path (under `data/demo_<n>/`) | Shape | Description |
|-------------------------------|--------|--------------|
| `actions` | `(T, 7)` | Commanded actions per timestep: 6D relative pose (position + orientation delta) + 1D gripper. |
| `initial_state/` | (groups below) | State at the start of the trajectory (`t=0`). |
| `obs/` | (datasets below) | Observations per timestep; aligned with `actions` (same `T`). |
| `states/` | (groups below) | Full simulation state per timestep (articulation + rigid objects). |

### `initial_state/`

One timestep only (shape `(1, …)` for each dataset):

- **`initial_state/articulation/robot/`**
  - `joint_position`: `(1, 9)` — 7 arm joints + 2 gripper.
  - `joint_velocity`: `(1, 9)`.
  - `root_pose`: `(1, 7)` — position (3) + quaternion (4).
  - `root_velocity`: `(1, 6)` — linear (3) + angular (3).

- **`initial_state/rigid_object/cube_1/`**, **`cube_2/`**, **`cube_3/`**
  - `root_pose`: `(1, 7)` — position (3) + quaternion (4).
  - `root_velocity`: `(1, 6)`.

### `obs/`

All have shape `(T, dim)` with the same `T` as `actions`:

| Dataset | Shape | Description |
|---------|--------|--------------|
| `actions` | `(T, 7)` | Same as top-level `actions` (last applied action). |
| `cube_positions` | `(T, 9)` | 3 cubes × 3D position (x,y,z) in world frame. |
| `cube_orientations` | `(T, 12)` | 3 cubes × quaternion (4) in world frame. |
| `eef_pos` | `(T, 3)` | End-effector position. |
| `eef_quat` | `(T, 4)` | End-effector orientation (quaternion). |
| `gripper_pos` | `(T, 2)` | Gripper state (e.g. left/right finger positions). |
| `joint_pos` | `(T, 9)` | Joint positions (9 DoF). |
| `joint_vel` | `(T, 9)` | Joint velocities. |
| `object` | `(T, 39)` | Object-centric observation vector. |

### `states/`

Full state per timestep; same structure as `initial_state/` but with time dimension `T` instead of 1:

- **`states/articulation/robot/`**
  - `joint_position`, `joint_velocity`: `(T, 9)`.
  - `root_pose`: `(T, 7)`.
  - `root_velocity`: `(T, 6)`.

- **`states/rigid_object/cube_1/`**, **`cube_2/`**, **`cube_3/`**
  - `root_pose`: `(T, 7)`.
  - `root_velocity`: `(T, 6)`.

## Inspecting a file

From the project root (or from `isaac_envs`):

```bash
python scripts/print_hdf5_structure.py
```

This uses the default path `isaac_envs/data/franka_stack_dataset.hdf5` (relative to the project root). To use another file:

```bash
python scripts/print_hdf5_structure.py /path/to/file.hdf5
```

Use `--full` to print the full tree for every demo instead of a template plus summary.

## Example summary output

```
File: .../isaac_envs/data/franka_stack_dataset.hdf5
Root keys: ['data']

data/: 10 demos
  Trajectory lengths: min=179, max=247, total_steps=2163

Structure (template: data/demo_0 and its subgroups):
...
All demo keys and lengths:
  demo_0: 236 steps
  demo_1: 233 steps
  ...
```
