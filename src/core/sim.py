from __future__ import annotations

import numpy as np

from src.core.env import EnvState, step

try:
    from src.render.pygame_renderer import process_window_events
except Exception:  # pragma: no cover - optional during non-visual runs
    def process_window_events() -> None:
        return


def sample_action(rng: np.random.Generator, env: EnvState, state: int, policy: np.ndarray) -> int:
    return int(rng.choice(env.actions, p=policy[state]))


def run_episode(
    env: EnvState,
    start_state: int,
    policy: np.ndarray,
    seed: int = 0,
    max_steps: int = 10_000,
) -> tuple[list[int], list[float]]:
    rng = np.random.default_rng(seed + int(start_state))
    state = int(start_state)
    state_history = [state]
    reward_history: list[float] = []

    for _ in range(max_steps):
        process_window_events()
        if state == env.goal_state:
            break
        action = sample_action(rng, env, state, policy)
        ns, rew = step(env, state, action)
        state_history.append(ns)
        reward_history.append(rew)
        state = ns

    return state_history, reward_history


def verify_policy_sampling(
    env: EnvState,
    policy: np.ndarray,
    samples_per_state: int = 10_000,
    seed: int = 0,
) -> tuple[list[dict], float]:
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    rows: list[dict] = []
    for s in range(env.num_states):
        process_window_events()
        counts = np.zeros(len(env.actions), dtype=float)
        for _ in range(samples_per_state):
            # Keep window responsive during heavy Monte Carlo sampling.
            if (_ % 250) == 0:
                process_window_events()
            a = int(rng.choice(env.actions, p=policy[s]))
            counts[a] += 1
        empirical = counts / samples_per_state
        diff = np.abs(empirical - policy[s])
        max_diff = max(max_diff, float(diff.max()))
        for a in env.actions:
            rows.append(
                {
                    "state": s,
                    "action": int(a),
                    "action_name": env.action_names[int(a)],
                    "policy_prob": float(policy[s, a]),
                    "empirical_prob": float(empirical[a]),
                    "abs_diff": float(diff[a]),
                }
            )
    return rows, max_diff
