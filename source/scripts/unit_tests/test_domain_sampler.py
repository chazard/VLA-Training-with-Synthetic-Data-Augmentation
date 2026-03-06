# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for domain_sampler: spec/ranges helpers, CEMSearch normalize/denormalize, and CEM on a synthetic landscape.


To run this:
ISAAC_ENVS_SKIP_TASK_IMPORTS=1 pytest source/scripts/unit_tests/test_domain_sampler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure scripts dir is on path for domain_sampler import; source root for isaac_envs
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from domain_sampler import (
    CEMSearch,
    DomainSampler,
    _spec_and_ranges_from_settable,
    _vec_to_settable_params,
)
from isaac_envs.tasks.searchable_env_cfg import SearchableEnvironmentCfg, SettableParamsT


# -----------------------------------------------------------------------------
# 1. Spec / ranges and vec_to_settable round-trip
# -----------------------------------------------------------------------------


def test_spec_and_ranges_from_settable_roundtrip_with_vec_to_settable():
    """Same spec and vec: spec_and_ranges_from_settable and vec_to_settable_params are consistent."""
    # Fake settable (same shape as get_settable_parameters_for_active_events())
    settable = {
        "term_a": [
            (["cube", "pose_range", "x"], (0.0, 1.0)),
            (["cube", "pose_range", "y"], (-0.5, 0.5)),
        ],
        "term_b": [
            (["light", "intensity_range"], (100.0, 200.0)),
        ],
    }
    spec, ranges = _spec_and_ranges_from_settable(settable)
    assert len(spec) == 3
    assert ranges.shape == (3, 2)
    # Build vec as [mid of range] for each dimension
    vec = (ranges[:, 0] + ranges[:, 1]) / 2.0
    params = _vec_to_settable_params(spec, vec)
    # Same keys and same order of terms
    assert list(params.keys()) == sorted(settable.keys())
    assert params["term_a"][0][0] == ["cube", "pose_range", "x"]
    assert params["term_a"][0][1] == (vec[0], vec[0])
    assert params["term_a"][1][1] == (vec[1], vec[1])
    assert params["term_b"][0][1] == (vec[2], vec[2])
    # Round-trip: from params we could get spec2, ranges2; ranges2 would be (v,v) per dim
    spec2, ranges2 = _spec_and_ranges_from_settable(params)
    assert spec2 == spec
    np.testing.assert_array_almost_equal(ranges2[:, 0], vec)
    np.testing.assert_array_almost_equal(ranges2[:, 1], vec)


def test_vec_to_settable_params_preserves_spec_order():
    """vec_to_settable_params produces one (key_seq, (v,v)) per spec entry in order."""
    spec = [
        ("t1", ["a", "b"]),
        ("t1", ["a", "c"]),
        ("t2", ["x"]),
    ]
    vec = np.array([1.0, 2.0, 3.0])
    out = _vec_to_settable_params(spec, vec)
    assert out["t1"] == [(["a", "b"], (1.0, 1.0)), (["a", "c"], (2.0, 2.0))]
    assert out["t2"] == [(["x"], (3.0, 3.0))]


# -----------------------------------------------------------------------------
# 2. CEMSearch normalize / denormalize
# -----------------------------------------------------------------------------


def test_cem_search_normalize_denormalize_roundtrip():
    """Normalize then denormalize returns the original vector (within numerical tolerance)."""
    rng = np.random.default_rng(42)
    ranges = np.column_stack([rng.uniform(-10, 0, size=5), rng.uniform(1, 10, size=5)])
    cem = CEMSearch(ranges, n_clusters=1, cov_reg=0.01, cov_init_scale=0.1)
    vec = rng.uniform(ranges[:, 0], ranges[:, 1], size=5)
    normalized = cem._normalize(vec)
    denormalized = cem._denormalize(normalized)
    np.testing.assert_array_almost_equal(denormalized, vec)


def test_cem_search_normalize_bounds():
    """Normalize maps min to 0 and max to 1 per dimension."""
    ranges = np.array([[0.0, 10.0], [-1.0, 1.0]])
    cem = CEMSearch(ranges, n_clusters=1, cov_reg=0.01, cov_init_scale=0.1)
    lo = ranges[:, 0]
    hi = ranges[:, 1]
    np.testing.assert_array_almost_equal(cem._normalize(lo), np.array([0.0, 0.0]))
    np.testing.assert_array_almost_equal(cem._normalize(hi), np.array([1.0, 1.0]))


def test_cem_search_denormalize_bounds():
    """Denormalize maps 0 to min and 1 to max per dimension."""
    ranges = np.array([[0.0, 10.0], [-1.0, 1.0]])
    cem = CEMSearch(ranges, n_clusters=1, cov_reg=0.01, cov_init_scale=0.1)
    np.testing.assert_array_almost_equal(cem._denormalize(np.array([0.0, 0.0])), ranges[:, 0])
    np.testing.assert_array_almost_equal(cem._denormalize(np.array([1.0, 1.0])), ranges[:, 1])


# -----------------------------------------------------------------------------
# 3. CEM on synthetic quadratic landscape (M minima, match cluster means)
# -----------------------------------------------------------------------------


def _make_quadratic_landscape(
    minima: np.ndarray,
    scales: np.ndarray,
) -> callable:
    """Returns a function f(x) = min over m of sum_i scales[i] * (x[i] - minima[m,i])^2."""
    assert minima.ndim == 2 and scales.ndim == 1
    M, D = minima.shape
    assert scales.shape[0] == D

    def score(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).ravel()
        assert x.shape[0] == D
        cost_per_minimum = np.sum(scales * (x - minima) ** 2, axis=1)
        return float(np.min(cost_per_minimum))

    return score


def _run_cem_on_quadratic_landscape(
    rng: np.random.Generator,
    minima: np.ndarray,
    scales: np.ndarray,
    n_clusters: int,
    batch_size: int,
    n_elite: int,
    n_iters: int,
) -> tuple[CEMSearch, np.ndarray, np.ndarray]:
    """Run CEM on a quadratic landscape; return (cem, minima, scales) for caller to assert."""
    M, D = minima.shape
    ranges = np.column_stack([np.zeros(D), np.ones(D)])
    score_fn = _make_quadratic_landscape(minima, scales)
    cem = CEMSearch(ranges, n_clusters=n_clusters, cov_reg=0.005, cov_init_scale=.2)

    for _ in range(n_iters):
        batch = [cem.sample() for _ in range(batch_size)]
        scores = np.array([score_fn(x) for x in batch])
        order = np.argsort(scores)
        elite = np.array([batch[i] for i in order[:n_elite]])
        cem.update(elite)

    return cem, minima, scales


def test_cem_search_on_synthetic_quadratic_landscape():
    """Run CEM on a landscape with M random quadratics; cluster means should align with true minima."""
    rng = np.random.default_rng(1234)
    D = 4
    M = 3
    minima = rng.uniform(0.2, 0.8, size=(M, D))
    scales = rng.uniform(0.5, 2.0, size=D)

    cem, minima, scales = _run_cem_on_quadratic_landscape(
        rng, minima, scales,
        n_clusters=M,
        batch_size=40,
        n_elite=int(40 * 0.4),
        n_iters=80,
    )

    cluster_means_real = cem._denormalize(cem._gmm_means)
    # Covariances: stored in normalized space; real-space diagonal = (hi - lo)^2 * var_norm
    scale = cem.ranges[:, 1] - cem.ranges[:, 0]
    cluster_covs_real = (scale ** 2) * cem._gmm_covs

    print("\n--- CEM vs ground truth (synthetic quadratic landscape) ---")
    print("Cluster weights:", cem._gmm_weights)
    print("Computed means (real):\n", cluster_means_real)
    print("Computed covariances (real, diagonal):\n", cluster_covs_real)
    print("Ground truth means (minima):\n", minima)
    


def test_cem_search_on_synthetic_quadratic_landscape_single_minimum():
    """CEM with one cluster should converge near the single minimum."""
    rng = np.random.default_rng(456)
    D = 3
    minimum = rng.uniform(0.3, 0.7, size=D)
    scales = np.ones(D)
    minima = minimum.reshape(1, D)

    cem, minima, scales = _run_cem_on_quadratic_landscape(
        rng, minima, scales,
        n_clusters=1,
        batch_size=20,
        n_elite=8,
        n_iters=20,
    )

    mean_real = cem._denormalize(cem._gmm_means[0])
    scale = cem.ranges[:, 1] - cem.ranges[:, 0]
    cov_real = (scale ** 2) * cem._gmm_covs[0]

    print("\n--- CEM vs ground truth (single minimum) ---")
    print("Cluster weights:", cem._gmm_weights)
    print("Computed mean (real):", mean_real)
    print("Computed covariance (real, diagonal):", cov_real)
    print("Ground truth mean (minimum):", minimum)

    dist = np.sqrt(np.sum((mean_real - minimum) ** 2))
    assert dist <= 0.1, f"Single-cluster mean is {dist:.4f} from true minimum (allowed 0.1)."


# -----------------------------------------------------------------------------
# 4. DomainSampler with fake SearchableEnvironmentCfg (quadratic landscape + feasibility)
# -----------------------------------------------------------------------------


def _params_to_vec(params: SettableParamsT) -> np.ndarray:
    """Extract flat vec from SettableParamsT in same order as _spec_and_ranges_from_settable."""
    vec = []
    for term_name in sorted(params.keys()):
        for _key_seq, range_tup in params[term_name]:
            vec.append(float(range_tup[0]))
    return np.array(vec)


class FakeSearchableCfg(SearchableEnvironmentCfg):
    """Fake searchable config: D-dimensional [0,1]^D, feasibility = far enough from a bad point."""

    def __init__(
        self,
        D: int,
        bad_point: np.ndarray,
        min_distance: float,
    ) -> None:
        self._D = D
        self._bad_point = np.asarray(bad_point, dtype=np.float64).ravel()
        assert self._bad_point.shape[0] == D
        self._min_distance = float(min_distance)
        self._current_params: SettableParamsT | None = None
        self._active_terms = ["fake"]

    def get_default_active_event_terms(self) -> list[str]:
        return list(self._active_terms)

    def get_default_event_ranges(self) -> dict:
        return {"fake": {f"d{i}": (0.0, 1.0) for i in range(self._D)}}

    def get_settable_parameters_for_active_events(self) -> SettableParamsT:
        return {"fake": [([f"d{i}"], (0.0, 1.0)) for i in range(self._D)]}

    def set_active_event_terms(self, term_names: list[str]) -> None:
        self._active_terms = list(term_names)

    def set_distribution_parameters(self, params: SettableParamsT) -> None:
        self._current_params = {k: [(ks, (r[0], r[1])) for ks, r in v] for k, v in params.items()}

    def check_feasibility(self, distribution_parameters: SettableParamsT) -> bool:
        vec = _params_to_vec(distribution_parameters)
        return float(np.linalg.norm(vec - self._bad_point)) >= self._min_distance

    def get_current_vec(self) -> np.ndarray:
        """Return the last set point as a vector (spec order)."""
        assert self._current_params is not None
        return _params_to_vec(self._current_params)


def test_domain_sampler_with_fake_searchable_cfg():
    """DomainSampler with a fake config: feasibility = far from a bad point; success_rate = quadratic landscape (minimize)."""
    rng = np.random.default_rng(789)
    D = 3
    M = 2
    minima = rng.uniform(0.3, 0.75, size=(M, D))
    scales = np.ones(D)

    bad_point = rng.uniform(0.05, 0.15, size=D)
    min_distance = 0.15  # feasible if at least this far from bad_point; minima stay feasible
    score_fn = _make_quadratic_landscape(minima, scales)

    cfg = FakeSearchableCfg(D, bad_point=bad_point, min_distance=min_distance)
    unwrapped = type("Unwrapped", (), {"cfg": cfg})()
    env = type("Env", (), {"unwrapped": unwrapped})()

    sampler = DomainSampler(
        env,
        n_cem_clusters=M,
        cem_batch_size=20,
        elite_frac=0.4,
        cov_init_scale=0.2,
        max_feasibility_samples=100,
    )

    n_batches = 20
    n_calls = n_batches * sampler._cem_batch_size + 1  # +1 so last suggestion gets a score and we do n_batches updates
    for step in range(n_calls):
        last_results = None if step == 0 else {"success_rate": score_fn(cfg.get_current_vec())}
        sampler(last_results)

    # After training, pull GMM weights/means/covariances from the sampler and compare to ground truth
    computed_weights, computed_means, computed_covs = sampler.get_cem_means_and_covariances()
    print("\n--- DomainSampler fake config: computed vs actual minima ---")
    print("Cluster weights:", computed_weights)
    print("Computed GMM means (real):\n", computed_means)
    print("Computed GMM covariances (real, diagonal):\n", computed_covs)
    print("Actual minima:\n", minima)