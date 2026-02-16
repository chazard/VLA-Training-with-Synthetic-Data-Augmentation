# Custom Stack Cube Environment — Agent Interface

This document describes how to run the custom stack cube visuomotor simulation and how to attach a custom agent (policy) that consumes observations and outputs actions.

## Running the simulation

1. **List registered envs** (includes the custom stack cube):
   ```bash
   python scripts/list_envs.py
   python scripts/list_envs.py --keyword Stack
   ```

2. **Run with a built-in agent** (zero or random actions):
   ```bash
   python scripts/custom_agent.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 4 --agent zero
   python scripts/custom_agent.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 4 --agent random
   ```

Use `PATH_TO_isaaclab.sh -p` (or your Isaac Lab Python) if Isaac Lab is not on your default `python`.

## Environment contract

- **Gym API**: Standard `gymnasium` vectorized env: `reset()` → `(obs, info)`, `step(actions)` → `(obs, reward, terminated, truncated, info)`.
- **Device**: All tensors (obs, actions) are on the env device; use `env.unwrapped.device` (e.g. `cuda:0`).
- **Batching**: Observations and actions are batched: first dimension is `num_envs`.

## Observation space

The env uses **observation groups**. The main group for a policy is **`"policy"`**.

After `obs, _ = env.reset()` or `obs, ... = env.step(actions)`:

- `obs` is a **nested dict**: `obs["policy"]` is the group used for the policy.
- `obs["policy"]` is a **dict** of tensors (no concatenation by default in the visuomotor config). Typical keys:

| Key              | Shape (per env)   | Description |
|------------------|-------------------|--------------|
| `table_cam`      | (H, W, 3)         | RGB image from fixed table camera (e.g. 200×200×3). |
| `wrist_cam`      | (H, W, 3)         | RGB image from wrist-mounted camera. |
| `joint_pos`      | (joint_dim,)      | Joint positions (relative to default). |
| `joint_vel`      | (joint_dim,)      | Joint velocities. |
| `object`         | (object_dim,)     | Object-centric observation. |
| `cube_positions` | (9,)              | 3 cubes × 3D position in world. |
| `cube_orientations` | (12,)           | 3 cubes × quat (4) in world. |
| `eef_pos`        | (3,)              | End-effector position. |
| `eef_quat`       | (4,)              | End-effector orientation (quat). |
| `gripper_pos`    | (1,)              | Gripper open/close. |
| `actions`        | (action_dim,)     | Last applied action. |

Exact shapes come from the config (e.g. camera 200×200). Inspect at runtime:

```python
obs, _ = env.reset()
for k, v in obs["policy"].items():
    print(k, v.shape)
```

## Action space

- **Shape**: `(num_envs, action_dim)`. `action_dim` is the sum of all action term dimensions (e.g. ~7: 6D relative pose + 1D gripper).
- **Semantics**: Differential IK (relative pose) + binary gripper:
  - First 6: relative end-effector pose change (position + orientation delta), scaled by config.
  - Last 1: gripper open/close (e.g. -1 open, +1 close).
- **Range**: Unbounded Box; the env and IK controller apply scaling/clipping internally.

All actions must be **torch tensors** on `env.unwrapped.device`.

## Implementing a custom agent

1. **Use the run script**  
   Edit `scripts/custom_agent.py` and replace the logic inside `custom_agent()` with your policy:
   - Input: `obs` (dict with at least `obs["policy"]`), `env`, and any extra args.
   - Output: `actions` tensor of shape `(num_envs, action_dim)` on `env.unwrapped.device`.

2. **Minimal loop** (without the script):
   ```python
   import gymnasium as gym
   import torch
   import isaac_envs.tasks  # noqa: F401
   from isaaclab_tasks.utils import parse_env_cfg

   task = "Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0"
   env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=4)
   env = gym.make(task, cfg=env_cfg)
   obs, _ = env.reset()
   while True:  # or your condition
       with torch.inference_mode():
           actions = your_policy(obs["policy"])  # (num_envs, action_dim)
       obs, rewards, dones, truncated, infos = env.step(actions)
   env.close()
   ```

3. **Image-based policy**  
   Use `obs["policy"]["table_cam"]` and/or `obs["policy"]["wrist_cam"]` (and optionally state keys) as inputs to your vision or visuomotor model; output a tensor of shape `(num_envs, action_dim)`.

4. **Config and event terms**  
   To change randomization (cubes, lights, cameras, etc.) or which events run at reset, adjust `CustomCubeStackEnvCfg` (e.g. `active_event_terms`, `set_distribution_parameters`) before creating the env; see `stack_cube_wrapper_cfg.py`.
