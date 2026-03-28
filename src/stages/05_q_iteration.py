from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.env import env_from_jsonable
from src.core.rl import make_greedy_policy, run_q_iteration
from src.io.artifacts import load_json, load_npy, save_json, save_npy, stage_figure_dir
from src.render.pygame_renderer import draw_grid, draw_line_plot, draw_matrix_heatmap


def run(results_dir: Path) -> dict:
    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    policy = load_npy(results_dir / "data" / "policy_init.npy")
    v_ref = load_npy(results_dir / "data" / "V.npy")
    gamma = float(env_payload["gamma"])

    p_action, v_q, q, conv, _, _ = run_q_iteration(env, policy, gamma=gamma, num_iter=1000)

    save_npy(results_dir / "data" / "V_q.npy", v_q)
    save_npy(results_dir / "data" / "Q.npy", q)

    for a in env.actions:
        save_npy(results_dir / "data" / f"P_action_{int(a)}.npy", p_action[int(a)])

    greedy = make_greedy_policy(env, q)
    save_npy(results_dir / "data" / "policy_greedy.npy", greedy)

    fig_dir = stage_figure_dir(results_dir, "05_q")
    draw_matrix_heatmap(q, fig_dir / "q_heatmap.png", "Q-function Heatmap", max_cells=40)
    best_actions = np.argmax(q, axis=1)
    draw_grid(env, fig_dir / "greedy_policy_grid.png", "Greedy Policy from Q", best_actions=best_actions, cell_values=v_q)
    draw_line_plot([np.asarray(conv)], ["max |dQ|"], fig_dir / "q_convergence.png", "Q Iteration Convergence", "delta", log_y=True)

    summary = {
        "max_abs_diff_Vq_vs_V": float(np.max(np.abs(v_q - v_ref))),
        "final_q_delta": float(conv[-1]),
    }
    save_json(results_dir / "data" / "stage05_summary.json", summary)
    return summary
