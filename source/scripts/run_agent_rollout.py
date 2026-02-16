# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the custom stack cube visuomotor env with a pluggable agent.

Use this script to run the simulation and attach a custom agent (see docs/AGENT_INTERFACE.md).
Demo-replay comparison (--demo_replay_comparison): two envs in lockstep, model vs replayed demo; see docs/GROOT_POLICY_CLIENT_SETUP.md § Comparison rollout.

Example:
  python scripts/run_agent_rollout.py --task Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0 --num_envs 4 --agent zero
  python scripts/run_agent_rollout.py --agent groot --demo_replay_comparison --hdf5-path /path/to/data.hdf5 --demo-index 0 --start-step 0
"""

import argparse
import sys
from pathlib import Path

# Add isaac_envs package source so "import isaac_envs.tasks" works when run from repo root.
_ISAAC_ENVS_SOURCE = Path(__file__).resolve().parent.parent
if _ISAAC_ENVS_SOURCE.exists() and str(_ISAAC_ENVS_SOURCE) not in sys.path:
    sys.path.insert(0, str(_ISAAC_ENVS_SOURCE))
# Add groot so "groot" agent can import gr00t.policy.server_client (optional; only needed for --agent groot).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GROOT_ROOT = _REPO_ROOT / "third_party" / "groot"
if _GROOT_ROOT.exists() and str(_GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GROOT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run custom stack cube env with a custom agent.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0",
)
parser.add_argument("--agent", type=str, choices=["zero", "random", "groot"], default="zero")
parser.add_argument(
    "--max_steps",
    type=int,
    default=300,
    help="Reset env every this many steps (and log).",
)
#Arguments for side by side demo replay comparison against the agent
parser.add_argument("--demo_replay_comparison", action="store_true", help="Run model vs replay (env 0 vs env 1); requires --hdf5-path, --agent groot.")
parser.add_argument("--hdf5-path", type=str, default=None, help="Path to Franka stack HDF5 (required when --demo_replay_comparison).")
parser.add_argument("--demo-index", type=int, default=0, help="Demo index in dataset (e.g. 0 = demo_0). Used when --demo_replay_comparison.")
parser.add_argument("--start-step", type=int, default=0, help="Starting step in demo (0 = start). Used when --demo_replay_comparison.")
parser.add_argument("--model-port", type=int, default=5555, help="Model policy server port when --demo_replay_comparison.")
parser.add_argument("--replay-port", type=int, default=5556, help="Replay policy server port when --demo_replay_comparison.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaac_envs.tasks  # noqa: F401

from agent import make_agent, make_groot_comparison_agents
from hdf5_state_loader import load_demo_state_from_hdf5


def _slice_obs_for_env(obs: dict, env_index: int) -> dict:
    """Observation for a single env (batch size 1) for policy agents."""
    policy = obs.get("policy", obs)
    return {
        "policy": {
            k: v[env_index : env_index + 1] if hasattr(v, "__getitem__") else v
            for k, v in policy.items()
        }
    }


def main():
    use_demo_replay_comparison = args_cli.demo_replay_comparison
    if use_demo_replay_comparison:
        if not args_cli.hdf5_path:
            raise SystemExit("--demo_replay_comparison requires --hdf5-path.")
        if args_cli.agent != "groot":
            raise SystemExit("--demo_replay_comparison requires --agent groot.")
        num_envs = 2
    else:
        num_envs = args_cli.num_envs

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    if use_demo_replay_comparison:
        model_agent, replay_agent = make_groot_comparison_agents(
            env, model_port=args_cli.model_port, replay_port=args_cli.replay_port
        )
        replay_agent.policy.reset(
            options={"episode_index": args_cli.demo_index, "step_index": args_cli.start_step}
        )
        hdf5_path = Path(args_cli.hdf5_path)
        if not hdf5_path.exists():
            raise SystemExit(f"HDF5 not found: {hdf5_path}")
        loaded_state = load_demo_state_from_hdf5(
            hdf5_path, args_cli.demo_index, args_cli.start_step, device=device, num_envs=2
        )
        env.reset()
        obs, _ = env.unwrapped.reset_to(
            loaded_state, env_ids=torch.tensor([0, 1], device=device)
        )
        print(f"[INFO] Comparison: demo={args_cli.demo_index}, start_step={args_cli.start_step}. Env0=model, Env1=replay.")
    else:
        agent = make_agent(args_cli.agent, env)
        print("[INFO] Observation space:", env.observation_space)
        print("[INFO] Action space:", env.action_space)
        obs, _ = env.reset()

    print("[INFO] Starting simulation loop...")
    step = 0
    while simulation_app.is_running():
        step += 1
        if use_demo_replay_comparison:
            with torch.inference_mode():
                a0 = model_agent(_slice_obs_for_env(obs, 0))["action"]
                a1 = replay_agent(_slice_obs_for_env(obs, 1))["action"]
                actions = torch.cat([a0, a1], dim=0)
            obs, rewards, dones, truncated, infos = env.step(actions)
            if dones.any() or truncated.any():
                print(f"[INFO] Step {step} -- episode ended")
                break
        else:
            if step % args_cli.max_steps == 0:
                obs, _ = env.reset()
                print(f"[INFO] Step {step} -- Resetting")
            with torch.inference_mode():
                actions = agent(obs)["action"]
            obs, rewards, dones, truncated, infos = env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
