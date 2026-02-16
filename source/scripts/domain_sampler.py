# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial domain sampler: takes batch results (success/reward) and returns domain
parameters for the next round to lower success/reward (harder conditions).
Gets default event ranges and active event terms from the env config (see wrapper_cfg
get_default_event_ranges / get_default_active_event_terms) so the same sampler can
apply to future environments that expose that interface.
"""

from __future__ import annotations

from typing import Any


class DomainSampler:
    """Adversarial search: given last batch results, output domain parameters for the next
    round to reduce success rate / reward (harder domain). Requires the environment so
    it can read default event ranges and active event terms from the env's config
    (wrapper_cfg exposes get_default_event_ranges and get_default_active_event_terms).
    """

    def __init__(self, env: Any) -> None:
        """Store env and resolve default event ranges / active terms from its config."""
        self._env = env
        cfg = getattr(env.unwrapped, "cfg", None)
        assert cfg is not None and hasattr(cfg, "get_default_event_ranges") and hasattr(cfg, "get_default_active_event_terms")
        default_event_ranges = cfg.get_default_event_ranges()
        default_active_event_terms = cfg.get_default_active_event_terms()

        self.initial_distributions = {}
        for event in default_active_event_terms:
            self.initial_distributions[event] = default_event_ranges[event]
    
    def __call__(self, last_batch_results: dict[str, Any] | None) -> dict[str, Any]:
        """Compute domain parameters for the next round from the last batch.

        Args:
            last_batch_results: From the previous round. Expected keys:
                - success_rate: float in [0, 1]
                - mean_reward: float
                - num_envs: int
                None on the first round.

        Returns:
            Dict to apply to env config:
                - distribution_parameters: dict for set_distribution_parameters(...)
                - active_event_terms: optional list for set_active_event_terms(...)
        """
        # TODO: Implement cross-entropy method (CEM) search to optimize domain parameters
        # for lowering success rate / reward (adversarial conditions). Use last_batch_results
        # to rank or select candidate params and update the sampling distribution each round.
        return {
            "distribution_parameters": {},
        }
