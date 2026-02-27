#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the stack cube environment: run N rounds (num_rounds * num_envs episodes), report success rate and mean reward.
Use --headless for no GUI.

If we run this normally (not the adversarial domain search), then each round will spawn
domain randomized environments according to the default event distributions set for that environment
(recall that lighting is a global event, so all environments will have the same lighting).

If we run this with the adversarial domain search, then each round will spawn domain randomized environments according to the domain sampler's 
returned distribution parameters, which are not ranges, i.e. all of the environments in a given round will have the 
exact same domain parameters and the only randomness will come from the policy execution randomness (i.e. the diffusion head).
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ISAAC_ENVS_SOURCE = _SCRIPT_DIR.parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_GROOT_ROOT = _REPO_ROOT / "third_party" / "groot"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if _ISAAC_ENVS_SOURCE.exists() and str(_ISAAC_ENVS_SOURCE) not in sys.path:
    sys.path.insert(0, str(_ISAAC_ENVS_SOURCE))
if _GROOT_ROOT.exists() and str(_GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GROOT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate stack cube env. Use --headless for no GUI.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--num_rounds", type=int, default=50, help="Number of evaluation rounds (each round runs num_envs episodes in parallel).")
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0")
parser.add_argument("--agent", type=str, choices=["zero", "random", "groot"], default="zero")
#Use --run_adversarial_domain_rand_search to use DomainSampler to search for domains with lower success
parser.add_argument("--run_adversarial_domain_rand_search", action="store_true", help="Use DomainSampler to adapt domain params each round to lower success/reward.")
parser.add_argument("--num_cem_clusters", type=int, default=1, help="Number of GMM components in CEM when using adversarial domain search (default: 2).")
#Use --headless for no GUI.
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import isaac_envs.tasks  # noqa: F401

from agent import make_agent
from domain_sampler import DomainSampler
from isaac_envs.tasks.searchable_env_cfg import SearchableEnvironmentCfg


def main():
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env = gym.make(args.task, cfg=env_cfg)
    agent = make_agent(args.agent, env)
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    max_steps = getattr(env.unwrapped, "max_episode_length", 1000)

    domain_sampler = (
        DomainSampler(env, n_cem_clusters=args.num_cem_clusters)
        if args.run_adversarial_domain_rand_search
        else None
    )
    if domain_sampler is not None and not isinstance(env_cfg, SearchableEnvironmentCfg):
        raise RuntimeError("Adversarial domain rand search requires a searchable env config (e.g. CustomCubeStackEnvCfg).")

    all_rewards = []
    all_success = []
    last_batch_results = None

    for round_idx in range(args.num_rounds):
        if domain_sampler is not None:
            domain_sampler(last_batch_results)
        obs, _ = env.reset()
        rewards = torch.zeros(num_envs, device=device)
        completed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        round_success = [False] * num_envs

        for _ in range(max_steps):
            if not simulation_app.is_running() or completed.all():
                break
            with torch.inference_mode():
                action = agent(obs)["action"]
            obs, r, term, trunc, infos = env.step(action)

            # Compute success rate and mean reward for the round
            rewards[~completed] += r[~completed]
            just_finished = (term | trunc) & ~completed
            completed |= term | trunc
            if just_finished.any() and infos:
                log = infos.get("log") or infos
                succ = log.get("success", log.get("is_success"))
                if succ is not None:
                    s = np.atleast_1d(succ.cpu().numpy() if hasattr(succ, "cpu") else succ)
                    for i in just_finished.nonzero(as_tuple=False).flatten().cpu().tolist():
                        if i < len(s):
                            round_success[i] = bool(s[i])

        all_rewards.extend(rewards.cpu().tolist())
        all_success.extend(round_success)

        if domain_sampler is not None:
            last_batch_results = {
                "success_rate": sum(round_success) / num_envs,
                "mean_reward": sum(all_rewards[-num_envs:]) / num_envs,
            }
            print(
                f"Round {round_idx + 1}/{args.num_rounds} success rate: {last_batch_results['success_rate']:.2%}, "
                f"mean reward: {last_batch_results['mean_reward']:.4f}"
            )

    env.close()

    mean_reward = np.mean(all_rewards)
    success_rate = np.mean(all_success)
    print(f"Rounds: {args.num_rounds}")
    print(f"Num envs: {num_envs}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Mean reward: {mean_reward:.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
