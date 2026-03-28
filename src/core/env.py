from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class EnvConfig:
    grid_h: int = 10
    grid_w: int = 10
    obstacle_count: int = 10
    gamma: float = 0.99
    seed: int = 42


@dataclass
class EnvState:
    grid_h: int
    grid_w: int
    num_states: int
    goal_state: int
    obstacle_states: np.ndarray
    rewards: np.ndarray
    actions: np.ndarray
    action_dr: np.ndarray
    action_dc: np.ndarray
    action_names: list[str]

    @property
    def obstacle_set(self) -> set[int]:
        return set(int(s) for s in self.obstacle_states)


def state_to_rc(s: int, grid_w: int) -> tuple[int, int]:
    return divmod(int(s), int(grid_w))


def rc_to_state(r: int, c: int, grid_w: int) -> int:
    return int(r) * int(grid_w) + int(c)


def all_free_states_reach_goal(
    goal_state: int,
    obstacle_states: Iterable[int],
    num_states: int,
    grid_h: int,
    grid_w: int,
    actions: np.ndarray,
    action_dr: np.ndarray,
    action_dc: np.ndarray,
) -> bool:
    obstacle_set = set(int(s) for s in obstacle_states)
    free_states = [s for s in range(num_states) if s not in obstacle_set]

    reverse_edges: dict[int, list[int]] = {s: [] for s in free_states}
    for s in free_states:
        r, c = state_to_rc(s, grid_w)
        for a in actions:
            nr = int(np.clip(r + action_dr[a], 0, grid_h - 1))
            nc = int(np.clip(c + action_dc[a], 0, grid_w - 1))
            ns = rc_to_state(nr, nc, grid_w)
            if ns in obstacle_set:
                ns = s
            reverse_edges[ns].append(s)

    seen = {int(goal_state)}
    q = deque([int(goal_state)])
    while q:
        cur = q.popleft()
        for prev in reverse_edges[cur]:
            if prev not in seen:
                seen.add(prev)
                q.append(prev)

    return set(free_states).issubset(seen)


def step(env: EnvState, state: int, action: int) -> tuple[int, float]:
    if state == env.goal_state:
        return env.goal_state, 0.0

    if state in env.obstacle_set:
        return state, float(env.rewards[state])

    r, c = state_to_rc(state, env.grid_w)
    nr = int(np.clip(r + env.action_dr[action], 0, env.grid_h - 1))
    nc = int(np.clip(c + env.action_dc[action], 0, env.grid_w - 1))
    next_state = rc_to_state(nr, nc, env.grid_w)

    if next_state in env.obstacle_set:
        next_state = state

    return next_state, float(env.rewards[state])


def initialize_policy(env: EnvState, rng: np.random.Generator) -> np.ndarray:
    policy = rng.dirichlet(np.ones(len(env.actions)), size=env.num_states)
    policy[env.goal_state] = np.array([1.0, 0.0, 0.0, 0.0])

    for s in range(env.num_states):
        if s == env.goal_state:
            continue
        if s in env.obstacle_set:
            policy[s] = np.array([1.0, 0.0, 0.0, 0.0])
            continue

        allowed = np.ones(len(env.actions), dtype=float)
        r, c = state_to_rc(s, env.grid_w)
        for a in env.actions:
            nr = int(np.clip(r + env.action_dr[a], 0, env.grid_h - 1))
            nc = int(np.clip(c + env.action_dc[a], 0, env.grid_w - 1))
            ns = rc_to_state(nr, nc, env.grid_w)
            if ns in env.obstacle_set:
                allowed[a] = 0.0

        policy[s] *= allowed
        row_sum = policy[s].sum()
        if row_sum <= 0:
            policy[s] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            policy[s] /= row_sum

    return policy


def make_environment(cfg: EnvConfig) -> tuple[EnvState, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    num_states = cfg.grid_h * cfg.grid_w
    goal_state = int(rng.integers(0, num_states))

    actions = np.array([0, 1, 2, 3])
    action_dr = np.array([-1, 1, 0, 0])
    action_dc = np.array([0, 0, -1, 1])
    action_names = ["Up", "Down", "Left", "Right"]

    while True:
        candidates = [s for s in range(num_states) if s != goal_state]
        sampled = rng.choice(candidates, size=cfg.obstacle_count, replace=False)
        if all_free_states_reach_goal(
            goal_state,
            sampled,
            num_states,
            cfg.grid_h,
            cfg.grid_w,
            actions,
            action_dr,
            action_dc,
        ):
            obstacle_states = np.array(sorted(int(s) for s in sampled), dtype=int)
            break

    rewards = np.full(num_states, -1.0)
    rewards[goal_state] = 0.0
    rewards[obstacle_states] = -1.0

    env = EnvState(
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        num_states=num_states,
        goal_state=goal_state,
        obstacle_states=obstacle_states,
        rewards=rewards,
        actions=actions,
        action_dr=action_dr,
        action_dc=action_dc,
        action_names=action_names,
    )

    policy = initialize_policy(env, rng)
    return env, policy


def env_to_jsonable(env: EnvState, seed: int, gamma: float) -> dict:
    return {
        "grid_h": env.grid_h,
        "grid_w": env.grid_w,
        "num_states": env.num_states,
        "goal_state": env.goal_state,
        "obstacle_states": [int(x) for x in env.obstacle_states.tolist()],
        "rewards": [float(x) for x in env.rewards.tolist()],
        "actions": [int(x) for x in env.actions.tolist()],
        "action_dr": [int(x) for x in env.action_dr.tolist()],
        "action_dc": [int(x) for x in env.action_dc.tolist()],
        "action_names": list(env.action_names),
        "seed": int(seed),
        "gamma": float(gamma),
    }


def env_from_jsonable(payload: dict) -> EnvState:
    return EnvState(
        grid_h=int(payload["grid_h"]),
        grid_w=int(payload["grid_w"]),
        num_states=int(payload["num_states"]),
        goal_state=int(payload["goal_state"]),
        obstacle_states=np.array(payload["obstacle_states"], dtype=int),
        rewards=np.array(payload["rewards"], dtype=float),
        actions=np.array(payload["actions"], dtype=int),
        action_dr=np.array(payload["action_dr"], dtype=int),
        action_dc=np.array(payload["action_dc"], dtype=int),
        action_names=list(payload["action_names"]),
    )
