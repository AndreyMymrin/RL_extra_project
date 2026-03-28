from __future__ import annotations

import numpy as np

from src.core.env import EnvState, step


def build_transition_matrix(env: EnvState, policy: np.ndarray) -> np.ndarray:
    p = np.zeros((env.num_states, env.num_states), dtype=float)
    for s in range(env.num_states):
        for a in env.actions:
            ns, _ = step(env, s, int(a))
            p[s, ns] += policy[s, a]
    return p


def value_iteration_step(v: np.ndarray, p: np.ndarray, rewards: np.ndarray, gamma: float) -> tuple[np.ndarray, float]:
    v_new = rewards + gamma * (p @ v)
    delta = float(np.max(np.abs(v_new - v)))
    return v_new, delta


def run_value_iteration(
    env: EnvState,
    policy: np.ndarray,
    gamma: float,
    num_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray, list[float], list[np.ndarray]]:
    p = build_transition_matrix(env, policy)
    v = np.zeros(env.num_states, dtype=float)
    conv = []
    history = [v.copy()]
    for _ in range(num_iter):
        v, delta = value_iteration_step(v, p, env.rewards, gamma)
        conv.append(delta)
        history.append(v.copy())
    return p, v, conv, history


def build_action_transition_matrices(env: EnvState) -> dict[int, np.ndarray]:
    p_action: dict[int, np.ndarray] = {}
    for a in env.actions:
        pa = np.zeros((env.num_states, env.num_states), dtype=float)
        for s in range(env.num_states):
            ns, _ = step(env, s, int(a))
            pa[s, ns] = 1.0
        p_action[int(a)] = pa
    return p_action


def q_iteration_step(
    v: np.ndarray,
    q: np.ndarray,
    p_action: dict[int, np.ndarray],
    policy: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    q_new = np.zeros_like(q)
    for a in actions:
        q_new[:, a] = rewards + gamma * (p_action[int(a)] @ v)
    v_new = np.sum(policy * q_new, axis=1)
    delta = float(np.max(np.abs(q_new - q)))
    return v_new, q_new, delta


def run_q_iteration(
    env: EnvState,
    policy: np.ndarray,
    gamma: float,
    num_iter: int = 1000,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[float], list[np.ndarray], list[np.ndarray]]:
    p_action = build_action_transition_matrices(env)
    v = np.zeros(env.num_states, dtype=float)
    q = np.zeros((env.num_states, len(env.actions)), dtype=float)

    conv = []
    q_hist = [q.copy()]
    v_hist = [v.copy()]
    for _ in range(num_iter):
        v, q, delta = q_iteration_step(v, q, p_action, policy, env.rewards, gamma, env.actions)
        conv.append(delta)
        q_hist.append(q.copy())
        v_hist.append(v.copy())

    return p_action, v, q, conv, v_hist, q_hist


def make_greedy_policy(env: EnvState, q: np.ndarray) -> np.ndarray:
    pi = np.zeros((env.num_states, len(env.actions)), dtype=float)
    for s in range(env.num_states):
        pi[s, int(np.argmax(q[s]))] = 1.0
    pi[env.goal_state] = np.array([1.0, 0.0, 0.0, 0.0])
    for s in env.obstacle_states:
        pi[int(s)] = np.array([1.0, 0.0, 0.0, 0.0])
    return pi


def soft_policy_update(env: EnvState, policy: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    target = make_greedy_policy(env, q)
    pi_new = (1 - tau) * policy + tau * target
    pi_new[env.goal_state] = np.array([1.0, 0.0, 0.0, 0.0])
    for s in env.obstacle_states:
        pi_new[int(s)] = np.array([1.0, 0.0, 0.0, 0.0])
    return pi_new


def run_soft_policy_improvement(
    env: EnvState,
    init_policy: np.ndarray,
    p_action: dict[int, np.ndarray],
    gamma: float,
    tau: float = 0.3,
    rounds: int = 20,
    q_iters_per_round: int = 300,
    eval_starts: list[int] | None = None,
    run_episode_fn=None,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    policy = init_policy.copy()
    v = np.zeros(env.num_states, dtype=float)
    q = np.zeros((env.num_states, len(env.actions)), dtype=float)
    round_conv: list[float] = []
    round_avg_steps: list[float] = []

    for _ in range(rounds):
        for _ in range(q_iters_per_round):
            v, q, delta = q_iteration_step(v, q, p_action, policy, env.rewards, gamma, env.actions)
        round_conv.append(delta)

        if eval_starts and run_episode_fn is not None:
            steps = []
            for s in eval_starts:
                sh, _ = run_episode_fn(env, s, policy)
                steps.append(len(sh) - 1)
            round_avg_steps.append(float(np.mean(steps)))
        else:
            round_avg_steps.append(float("nan"))

        policy = soft_policy_update(env, policy, q, tau=tau)

    return policy, q, round_conv, round_avg_steps
