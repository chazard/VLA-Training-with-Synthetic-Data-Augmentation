# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Agent interface for evaluation: observation -> dict with "action"."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


def _import_policy_client():
    """Lazy import so zero/random agents work without the groot repo on PYTHONPATH."""
    try:
        from gr00t.policy.server_client import PolicyClient
        return PolicyClient
    except ImportError as e:
        raise ImportError(
            "GrootFrankaAgent requires the groot repo on PYTHONPATH (e.g. project_root/third_party/groot). "
            "Install groot deps and add: sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'third_party' / 'groot'))"
        ) from e


class Agent(ABC):
    """Returns dict with at least "action": (num_envs, action_dim) on env device."""

    @abstractmethod
    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        pass


class ZeroAgent(Agent):
    def __init__(self, device: torch.device, num_envs: int, action_dim: int):
        self.device = device
        self.num_envs = num_envs
        self.action_dim = action_dim

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        action = torch.zeros(
            (self.num_envs, self.action_dim),
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        return {"action": action}


class RandomAgent(Agent):
    def __init__(self, device: torch.device, num_envs: int, action_dim: int):
        self.device = device
        self.num_envs = num_envs
        self.action_dim = action_dim

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        action = 2 * torch.rand((self.num_envs, self.action_dim), device=self.device) - 1
        return {"action": action}


class GrootFrankaAgent(Agent):
    """Client-side agent that calls a GR00T policy server. Handles multiple envs via one batched request.

    Env obs keys (Franka stack): table_cam, wrist_cam, eef_pos, eef_quat; optional "task" or "language".
    Policy expects observation.video (B,T,H,W,3), observation.state (B,T,D), observation.language (B,1).
    We map table_cam -> video[ego_view], wrist_cam -> video[wrist_view], eef_pos/eef_quat -> state,
    and send one batch with B=num_envs. Policy returns action (B, action_horizon, action_dim);
    we use the first step and return (num_envs, action_dim) on the env device.
    """

    # Default env key -> policy modality key (video/state). Override or extend if your server uses different names.
    DEFAULT_VIDEO_MAP = {"table_cam": "ego_view", "wrist_cam": "wrist_view"}
    DEFAULT_STATE_KEYS = ["eef_pos", "eef_quat"]
    DEFAULT_LANGUAGE_KEY = "task"
    DEFAULT_TASK_STR = "stack the cubes"

    def __init__(
        self,
        policy_client: Any,
        device: torch.device,
        num_envs: int,
        action_dim: int,
        task_str: str | None = None,
        video_map: dict[str, str] | None = None,
        state_keys: list[str] | None = None,
        language_key: str | None = None,
    ):
        self.policy = policy_client
        self.device = device
        self.num_envs = num_envs
        self.action_dim = action_dim
        self.task_str = task_str or self.DEFAULT_TASK_STR
        self.video_map = video_map or dict(self.DEFAULT_VIDEO_MAP)
        self.state_keys = state_keys or list(self.DEFAULT_STATE_KEYS)
        self.language_key = language_key or self.DEFAULT_LANGUAGE_KEY
        self._modality_config: dict[str, Any] | None = None

    def _get_modality_config(self) -> dict[str, Any]:
        if self._modality_config is None:
            self._modality_config = self.policy.get_modality_config()
        return self._modality_config

    def _obs_to_policy_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Build policy observation dict with B=num_envs and T from modality config.
        T is the observation history length (e.g. T=1 in franka_stack_groot_config: current frame only).
        We replicate the current frame when T>1 if no history buffer is available."""
        cfg = self._get_modality_config()
        B = self.num_envs

        # Video: (B, T, H, W, 3) uint8
        video_cfg = cfg.get("video")
        T_video = len(video_cfg.delta_indices) if video_cfg else 1
        video = {}
        for env_key, policy_key in self.video_map.items():
            if env_key not in obs:
                continue
            x = obs[env_key]
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            x = np.asarray(x, dtype=np.uint8)
            if x.ndim == 3:
                x = x[:, None, ...]  # (B,H,W,3) -> (B,1,H,W,3)
            elif x.ndim == 4 and x.shape[1] != T_video:
                x = np.repeat(x[:, :1], T_video, axis=1)  # repeat first frame to fill T
            video[policy_key] = x
        out = {"video": video}

        # State: (B, T, D) float32
        state_cfg = cfg.get("state")
        T_state = len(state_cfg.delta_indices) if state_cfg else 1
        state_blocks = []
        for k in self.state_keys:
            if k not in obs:
                continue
            x = obs[k]
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x[None, None, :]  # (D,) -> (1,1,D)
            elif x.ndim == 2:
                x = x[:, None, :]  # (B,D) -> (B,1,D)
            if x.shape[1] != T_state:
                x = np.repeat(x[:, :1], T_state, axis=1)
            state_blocks.append(x)
        if state_blocks:
            state_concat = np.concatenate(state_blocks, axis=-1)  # (B, T, D)
            state_keys_cfg = state_cfg.modality_keys if state_cfg else ["state"]
            out["state"] = {state_keys_cfg[0]: state_concat} if len(state_keys_cfg) == 1 else {k: state_blocks[i] for i, k in enumerate(state_keys_cfg)}

        # Language: (B, 1) list of lists
        lang_cfg = cfg.get("language")
        task_str = obs.get(self.language_key, self.task_str)
        if isinstance(task_str, (list, tuple)):
            tasks = list(task_str)[:B]
        else:
            tasks = [str(task_str)] * B
        lang_key = lang_cfg.modality_keys[0] if lang_cfg and lang_cfg.modality_keys else self.DEFAULT_LANGUAGE_KEY
        out["language"] = {lang_key: [[t] for t in tasks]}
        return out

    def _policy_action_to_env_action(self, action_dict: dict[str, Any]) -> torch.Tensor:
        """Take first action step and convert to (num_envs, action_dim) on env device."""
        cfg = self._get_modality_config()
        action_cfg = cfg.get("action")
        keys = action_cfg.modality_keys if action_cfg else list(action_dict.keys())
        if not keys:
            raise RuntimeError("Policy returned no action keys")
        # Single action stream: (B, horizon, D)
        first_key = keys[0]
        arr = action_dict[first_key]
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr).to(device=self.device, dtype=torch.get_default_dtype())
        else:
            arr = arr.to(self.device)
        step0 = arr[:, 0, :]  # (B, action_dim)
        if step0.shape[-1] != self.action_dim:
            raise ValueError(
                f"Policy action dim {step0.shape[-1]} != env action_dim {self.action_dim}. "
                "Check embodiment/modality config matches env."
            )
        return step0

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        policy_obs = self._obs_to_policy_obs(obs)
        action_dict, _ = self.policy.get_action(policy_obs)
        action = self._policy_action_to_env_action(action_dict)
        return {"action": action}


def make_agent(agent_type: str, env) -> Agent:
    """Build an Agent from agent_type ("zero", "random", or "groot") for the given gym env."""
    u = env.unwrapped
    n = u.num_envs
    d = u.single_action_space.shape[0]
    if agent_type == "zero":
        return ZeroAgent(u.device, n, d)
    if agent_type == "random":
        return RandomAgent(u.device, n, d)
    if agent_type == "groot":
        PolicyClient = _import_policy_client()
        client = PolicyClient(host="localhost", port=5555, strict=False)
        if not client.ping():
            raise RuntimeError("GR00T policy server not reachable at localhost:5555. Start it first.")
        return GrootFrankaAgent(client, u.device, n, d)
    raise ValueError(f"Unknown agent_type: {agent_type!r}")


def make_groot_comparison_agents(
    env,
    model_port: int = 5555,
    replay_port: int = 5556,
    host: str = "localhost",
) -> tuple[GrootFrankaAgent, GrootFrankaAgent]:
    """Build model and replay agents for comparison rollout (num_envs=1 each).

    Returns (model_agent, replay_agent). Call replay_agent.policy.reset(options={"episode_index": i, "step_index": k})
    before the first step to sync replay to the chosen demo/step.
    """
    PolicyClient = _import_policy_client()
    u = env.unwrapped
    d = u.single_action_space.shape[0]
    model_client = PolicyClient(host=host, port=model_port, strict=False)
    replay_client = PolicyClient(host=host, port=replay_port, strict=False)
    if not model_client.ping():
        raise RuntimeError(f"Model policy server not reachable at {host}:{model_port}. Start it first.")
    if not replay_client.ping():
        raise RuntimeError(f"Replay policy server not reachable at {host}:{replay_port}. Start it first.")
    model_agent = GrootFrankaAgent(model_client, u.device, 1, d)
    replay_agent = GrootFrankaAgent(replay_client, u.device, 1, d)
    return model_agent, replay_agent
