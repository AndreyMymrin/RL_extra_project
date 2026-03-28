from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.env import env_from_jsonable, step
from src.core.sim import verify_policy_sampling
from src.io.artifacts import load_json, load_npy, save_csv, save_json, stage_figure_dir
from src.render.pygame_renderer import draw_matrix_heatmap


def run(results_dir: Path) -> dict:
    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    policy = load_npy(results_dir / "data" / "policy_init.npy")

    rows = []
    blocked_checks = []
    for s in range(env.num_states):
        for a in env.actions:
            ns, rew = step(env, s, int(a))
            rows.append(
                {
                    "state": s,
                    "action": int(a),
                    "action_name": env.action_names[int(a)],
                    "next_state": int(ns),
                    "reward": float(rew),
                }
            )

            if s not in env.obstacle_set and s != env.goal_state:
                r, c = divmod(s, env.grid_w)
                nr = int(np.clip(r + env.action_dr[a], 0, env.grid_h - 1))
                nc = int(np.clip(c + env.action_dc[a], 0, env.grid_w - 1))
                candidate = nr * env.grid_w + nc
                if candidate in env.obstacle_set:
                    blocked_checks.append(int(ns == s))

    save_csv(results_dir / "tables" / "transition_table.csv", rows)

    sampling_rows, max_diff = verify_policy_sampling(env, policy, samples_per_state=10000, seed=int(env_payload["seed"]))
    save_csv(results_dir / "tables" / "policy_sampling_check.csv", sampling_rows)

    # Transition diagnostics matrix for visualization: deterministic next-state index per (s,a)
    diag = np.zeros((env.num_states, len(env.actions)))
    for s in range(env.num_states):
        for a in env.actions:
            diag[s, a] = step(env, s, int(a))[0]

    fig_dir = stage_figure_dir(results_dir, "02_dynamics")
    draw_matrix_heatmap(diag, fig_dir / "next_state_matrix.png", "Diagnostics: next_state(s,a)")

    summary = {
        "max_policy_sampling_diff": float(max_diff),
        "blocked_entry_checks_passed": bool(all(blocked_checks) if blocked_checks else True),
        "transition_rows": len(rows),
    }
    save_json(results_dir / "data" / "stage02_summary.json", summary)
    return summary
