from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.env import env_from_jsonable
from src.core.rl import build_action_transition_matrices, run_soft_policy_improvement
from src.core.sim import run_episode
from src.io.artifacts import load_json, load_npy, save_json, save_npy, stage_figure_dir
from src.render.pygame_renderer import draw_line_plot


def run(results_dir: Path) -> dict:
    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    policy = load_npy(results_dir / "data" / "policy_init.npy")
    gamma = float(env_payload["gamma"])

    free_starts = [s for s in range(env.num_states) if s != env.goal_state and s not in env.obstacle_set]
    eval_starts = free_starts[: min(20, len(free_starts))]

    p_action = build_action_transition_matrices(env)
    improved, q_final, round_conv, round_avg_steps = run_soft_policy_improvement(
        env,
        policy,
        p_action,
        gamma=gamma,
        tau=0.3,
        rounds=20,
        q_iters_per_round=300,
        eval_starts=eval_starts,
        run_episode_fn=lambda env_, s, pi: run_episode(env_, s, pi, seed=int(env_payload["seed"]), max_steps=5000),
    )

    save_npy(results_dir / "data" / "policy_improved.npy", improved)
    save_npy(results_dir / "data" / "Q_improved.npy", q_final)

    fig_dir = stage_figure_dir(results_dir, "06_improve")
    draw_line_plot(
        [np.asarray(round_avg_steps), np.asarray(round_conv)],
        ["avg steps", "q delta"],
        fig_dir / "improvement_curves.png",
        "Policy Improvement Curves",
        "metric",
        log_y=False,
    )

    summary = {
        "rounds": 20,
        "final_avg_steps": float(round_avg_steps[-1]),
        "final_conv": float(round_conv[-1]),
    }
    save_json(results_dir / "data" / "stage06_summary.json", summary)
    return summary
