#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the stack cube environment: run N episodes, report success rate and mean reward.
Use --headless for no GUI."""

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
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--task", type=str, default="Isaac-Stack-Cube-Franka-Visuomotor-Custom-v0")
parser.add_argument("--agent", type=str, choices=["zero", "random", "groot"], default="zero")
#Use --run_adversarial_domain_rand_search to use DomainSampler to search for domains with lower success
parser.add_argument("--run_adversarial_domain_rand_search", action="store_true", help="Use DomainSampler to adapt domain params each round to lower success/reward.")
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
    max_steps = getattr(env.unwrapped, "max_episode_length", 2000)

    domain_sampler = DomainSampler(env) if args.run_adversarial_domain_rand_search else None
    if domain_sampler is not None and not (hasattr(env_cfg, "set_distribution_parameters") and hasattr(env_cfg, "set_active_event_terms")):
        raise RuntimeError("Adversarial domain rand search requires env config with set_distribution_parameters and set_active_event_terms (e.g. CustomCubeStackEnvCfg).")

    all_rewards = []
    all_success = []
    num_rounds = (args.num_episodes + num_envs - 1) // num_envs
    last_batch_results = None

    for _ in range(num_rounds):
        if domain_sampler is not None:
            out = domain_sampler(last_batch_results)
            if out.get("distribution_parameters"):
                env_cfg.set_distribution_parameters(out["distribution_parameters"])
            if out.get("active_event_terms") is not None:
                env_cfg.set_active_event_terms(out["active_event_terms"])
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
                "success_rate": sum(all_success[-num_envs:]) / num_envs if num_envs else 0.0,
                "mean_reward": sum(all_rewards[-num_envs:]) / num_envs if num_envs else 0.0,
                "num_envs": num_envs,
            }

    env.close()

    n = min(len(all_rewards), args.num_episodes)
    mean_reward = sum(all_rewards[:n]) / n if n else 0.0
    success_rate = sum(all_success[:n]) / n if n else 0.0
    print(f"Episodes: {n}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Mean reward: {mean_reward:.4f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
