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

import numpy as np
from sklearn.mixture import GaussianMixture

from isaac_envs.tasks.searchable_env_cfg import SearchableEnvironmentCfg


def _spec_and_ranges_from_settable(
    settable: Any,
) -> tuple[list[tuple[str, list[str]]], np.ndarray]:
    """Build flat spec and per-dimension (min, max) ranges from get_settable_parameters_for_active_events() output."""
    spec: list[tuple[str, list[str]]] = []
    ranges_list: list[tuple[float, float]] = []
    for term_name in sorted(settable.keys()):
        for key_seq, range_tup in settable[term_name]:
            spec.append((term_name, list(key_seq)))
            ranges_list.append((float(range_tup[0]), float(range_tup[1])))
    return spec, np.array(ranges_list, dtype=np.float64)  # shape (D, 2)


def _vec_to_settable_params(
    spec: list[tuple[str, list[str]]], vec: np.ndarray
) -> dict[str, list[tuple[list[str], tuple[float, float]]]]:
    """Convert flat vec back to SettableParamsT (same format as get_settable_parameters_for_active_events)."""
    out: dict[str, list[tuple[list[str], tuple[float, float]]]] = {}
    for idx, (term_name, key_seq) in enumerate(spec):
        if idx >= len(vec):
            break
        v = float(vec[idx])
        if term_name not in out:
            out[term_name] = []
        out[term_name].append((list(key_seq), (v, v)))
    return out


class CEMSearch:
    """CEM with a GMM over a continuous vector, with input/output normalized to [0,1] per dimension.

    Takes an initial set of ranges (min, max) per dimension; initializes at the center of these
    ranges. All sampling and elite updates are done in normalized space, then denormalized for
    the returned vectors.
    """

    def __init__(
        self,
        ranges: np.ndarray,
        n_clusters: int,
        cov_reg: float,
        cov_init_scale: float,
        smoothing_alpha: float = 0.3,
    ) -> None:
        """Initialize search with random means in normalized [0,1]^D.

        Args:
            ranges: Shape (D, 2), each row is (min, max) for that dimension.
            n_clusters: Number of Gaussian components in the mixture.
            cov_reg: Regularization added to diagonal covariance.
            cov_init_scale: Initial diagonal std in normalized space (e.g. 0.1).
            smoothing_alpha: Mixing factor for smoothing step (1-alpha)*old + alpha*new.
        """
        self.ranges = np.asarray(ranges, dtype=np.float64)
        assert self.ranges.ndim == 2 and self.ranges.shape[1] == 2
        self.D = self.ranges.shape[0]
        assert n_clusters >= 1, "n_clusters must be >= 1"
        self._n_clusters = n_clusters
        self._cov_reg = cov_reg
        self._cov_init_scale = cov_init_scale
        assert 0.0 <= smoothing_alpha <= 1.0, "smoothing_alpha must be in [0, 1]"
        self._smoothing_alpha = smoothing_alpha

        self._gmm_weights = np.ones(self._n_clusters) / self._n_clusters
        self._gmm_means = np.random.uniform(0.0, 1.0, size=(self._n_clusters, self.D)).astype(
            np.float64
        )
        self._gmm_covs = np.full(
            (self._n_clusters, self.D), self._cov_init_scale ** 2, dtype=np.float64
        )

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        return (vec - lo) / (hi - lo + 1e-9)

    def _denormalize(self, vec_norm: np.ndarray) -> np.ndarray:
        lo, hi = self.ranges[:, 0], self.ranges[:, 1]
        return lo + vec_norm * (hi - lo)

    def sample(self) -> np.ndarray:
        """Sample a vector in real space (denormalized from [0,1]^D)."""
        k = np.random.choice(self._n_clusters, p=self._gmm_weights)
        mean = self._gmm_means[k]
        cov_diag = np.maximum(self._gmm_covs[k], self._cov_reg)
        vec_norm = mean + np.sqrt(cov_diag) * np.random.randn(self.D)
        vec_norm = np.clip(vec_norm, 0.0, 1.0)
        return self._denormalize(vec_norm)

    def update(self, elite_pts: np.ndarray) -> None:
        """Update GMM from elite points (in real space): EM fit then smoothing."""
        elite_pts = np.asarray(elite_pts, dtype=np.float64)
        if elite_pts.ndim == 1:
            elite_pts = elite_pts.reshape(1, -1)
        n_elite, D = elite_pts.shape
        assert n_elite >= self._n_clusters, (
            f"CEMSearch.update requires at least {self._n_clusters} elite points, got {n_elite}."
        )
        elite_norm = self._normalize(elite_pts)
        precisions_init = 1.0 / np.maximum(self._gmm_covs, self._cov_reg)
        gmm = GaussianMixture(
            n_components=self._n_clusters,
            covariance_type="diag",
            reg_covar=self._cov_reg,
            means_init=self._gmm_means,
            weights_init=self._gmm_weights,
            precisions_init=precisions_init,
            max_iter=20,
        )
        gmm.fit(elite_norm)

        pi_new = gmm.weights_
        means_new = gmm.means_
        covs_new = gmm.covariances_
        
        alpha = self._smoothing_alpha
        self._gmm_weights = (1 - alpha) * self._gmm_weights + alpha * pi_new
        self._gmm_weights = self._gmm_weights / self._gmm_weights.sum()

        self._gmm_means = (1 - alpha) * self._gmm_means + alpha * means_new
        #make sure gmm means stay in bounds
        self._gmm_means = np.clip(self._gmm_means, 0.0, 1.0)
        
        self._gmm_covs = (1 - alpha) * self._gmm_covs + alpha * covs_new
        #apply a variance floor to prevent premature collapse
        self._gmm_covs = np.maximum(self._gmm_covs, self._cov_reg)


class DomainSampler:
    """Uses a searchable env config to drive CEMSearch and map its output to SettableParamsT.

    Builds spec and ranges from get_settable_parameters_for_active_events(), creates
    CEMSearch at the center of those ranges, and each call feeds batch results in and
    returns the next distribution_parameters for the env.
    """

    def __init__(
        self,
        env: Any,
        n_cem_clusters: int = 1,
        cem_batch_size: int = 20,
        elite_frac: float = 0.4,
        cov_reg: float = 0.005,
        cov_init_scale: float = 0.2,
        max_feasibility_samples: int = 100,
    ) -> None:
        """Build spec and ranges from config; create CEMSearch at center of ranges.

        Args:
            env: Environment with unwrapped.cfg that is a SearchableEnvironmentCfg.
            n_cem_clusters: Number of GMM components in CEMSearch.
            cem_batch_size: Number of (params, success_rate) samples before updating.
            elite_frac: Fraction of batch used as elite (lowest success rate).
            cov_reg: Covariance regularization for CEMSearch.
            cov_init_scale: Initial spread in normalized space for CEMSearch.
            max_feasibility_samples: Max samples per call before raising if none feasible.
        """
        self._env = env
        self._cfg = env.unwrapped.cfg
        assert isinstance(self._cfg, SearchableEnvironmentCfg), (
            "DomainSampler requires a searchable env config (e.g. CustomCubeStackEnvCfg)."
        )
        self._active_terms = list(self._cfg.get_default_active_event_terms())
        settable = self._cfg.get_settable_parameters_for_active_events()
        self._spec, ranges = _spec_and_ranges_from_settable(settable)
        if len(self._spec) == 0:
            raise ValueError(
                "DomainSampler: get_settable_parameters_for_active_events() returned no parameters (D=0)."
            )
        self._cem_batch_size = max(2, cem_batch_size)
        self._n_elite = max(1, int(self._cem_batch_size * elite_frac))
        self._cem_search = CEMSearch(
            ranges,
            n_clusters=n_cem_clusters,
            cov_reg=cov_reg,
            cov_init_scale=cov_init_scale,
        )
        self._buffer = []
        self._last_suggested_vec = None
        self._max_feasibility_samples = max_feasibility_samples

    def __call__(self, last_batch_results: dict[str, Any] | None) -> None:
        """Feed batch results into CEM search and set the next ranges on the env config."""
        # Add the last suggested vector and its score to the buffer and then if we are at the batch size, do an iteration
        # of CEM to update the search distribution
        if last_batch_results is not None and self._last_suggested_vec is not None:
            success_rate = float(last_batch_results["success_rate"])
            self._buffer.append((self._last_suggested_vec.copy(), success_rate))

            if len(self._buffer) >= self._cem_batch_size:
                # Elite = lowest success rate (minimize success for adversarial search)
                self._buffer.sort(key=lambda x: x[1])
                elite_list = [vec for vec, _ in self._buffer[: self._n_elite]]
                elite_pts = np.array(elite_list)
                self._cem_search.update(elite_pts)
                self._buffer.clear()

        # Sample from CEM until we find a vector of feasible distribution parameters
        # We are just taking a simple rejection sampling approach here since we expect most configurations to be feasible
        rejected_count = 0
        for attempt in range(1, self._max_feasibility_samples + 1):
            next_vec = self._cem_search.sample()
            distribution_parameters = _vec_to_settable_params(self._spec, next_vec)
            if self._cfg.check_feasibility(distribution_parameters):
                break
            rejected_count += 1
        else:
            raise RuntimeError(
                f"DomainSampler: no feasible sample after {self._max_feasibility_samples} attempts."
            )
        if rejected_count > 0:
            pct = 100.0 * rejected_count / attempt
            print(f"DomainSampler: feasibility reject rate {pct:.1f}% ({rejected_count}/{attempt} samples)")
        self._last_suggested_vec = next_vec
        self._cfg.set_distribution_parameters(distribution_parameters)

    def get_cem_means_and_covariances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return current CEM GMM weights, means, and diagonal covariances in real (denormalized) space.

        Returns:
            weights: Shape (n_clusters,), mixture weights.
            means: Shape (n_clusters, D), cluster centers in the same space as the search ranges.
            covariances: Shape (n_clusters, D), diagonal variance per dimension (real space).
        """
        weights = self._cem_search._gmm_weights.copy()
        means_norm = self._cem_search._gmm_means
        covs_norm = self._cem_search._gmm_covs
        means_real = self._cem_search._denormalize(means_norm)
        scale = self._cem_search.ranges[:, 1] - self._cem_search.ranges[:, 0]
        covs_real = (scale ** 2) * covs_norm
        return weights, means_real, covs_real
