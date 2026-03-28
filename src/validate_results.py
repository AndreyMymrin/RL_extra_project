from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.core.env import env_from_jsonable
from src.io.artifacts import assert_file, load_json, load_npy


EXPECTED_FILES = [
    "data/env.json",
    "data/policy_init.npy",
    "data/P.npy",
    "data/V.npy",
    "data/Q.npy",
    "data/policy_improved.npy",
    "data/stage01_summary.json",
    "data/stage02_summary.json",
    "data/stage03_summary.json",
    "data/stage04_summary.json",
    "data/stage05_summary.json",
    "data/stage06_summary.json",
    "data/stage07_summary.json",
]


EXPECTED_FIGURES = [
    "figures/01_environment/grid_map.png",
    "figures/02_dynamics/next_state_matrix.png",
    "figures/03_episodes/trajectory_random_1.png",
    "figures/03_episodes/trajectory_random_2.png",
    "figures/04_value/transition_matrix_P.png",
    "figures/04_value/value_grid.png",
    "figures/04_value/value_by_state.png",
    "figures/04_value/value_convergence.png",
    "figures/05_q/q_heatmap.png",
    "figures/05_q/greedy_policy_grid.png",
    "figures/05_q/q_convergence.png",
    "figures/06_improve/improvement_curves.png",
    "figures/07_compare/comparison_board.png",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated RL artifacts")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    for rel in EXPECTED_FILES + EXPECTED_FIGURES:
        assert_file(results_dir / rel)

    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    policy = load_npy(results_dir / "data" / "policy_init.npy")
    p = load_npy(results_dir / "data" / "P.npy")

    assert len(env.obstacle_states) == 10, "obstacle count must be 10"
    assert env.goal_state not in env.obstacle_set, "goal cannot be obstacle"
    assert np.allclose(policy.sum(axis=1), 1.0), "policy rows must sum to 1"
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-8), "transition rows must sum to 1"

    s2 = load_json(results_dir / "data" / "stage02_summary.json")
    s7 = load_json(results_dir / "data" / "stage07_summary.json")
    assert bool(s2["blocked_entry_checks_passed"]), "blocked-entry checks failed"
    assert bool(s7["improved_not_worse"]), "improved policy is worse than random"

    print("Validation passed.")


if __name__ == "__main__":
    main()
