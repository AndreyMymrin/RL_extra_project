from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.env import env_from_jsonable
from src.core.rl import run_value_iteration
from src.io.artifacts import load_json, load_npy, save_json, save_npy, stage_figure_dir
from src.render.pygame_renderer import draw_bar_chart, draw_grid, draw_line_plot, draw_matrix_heatmap


def run(results_dir: Path) -> dict:
    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    policy = load_npy(results_dir / "data" / "policy_init.npy")
    gamma = float(env_payload["gamma"])

    p, v, conv, _ = run_value_iteration(env, policy, gamma=gamma, num_iter=300)

    save_npy(results_dir / "data" / "P.npy", p)
    save_npy(results_dir / "data" / "V.npy", v)

    fig_dir = stage_figure_dir(results_dir, "04_value")
    draw_matrix_heatmap(p, fig_dir / "transition_matrix_P.png", "Policy-induced Transition Matrix P", max_cells=40)
    draw_grid(env, fig_dir / "value_grid.png", "Value Function on Grid", cell_values=v)
    draw_bar_chart(v, fig_dir / "value_by_state.png", "Value Function by State Index", env.obstacle_set)
    draw_line_plot([np.asarray(conv)], ["max change"], fig_dir / "value_convergence.png", "Value Iteration Convergence", "delta", log_y=True)

    row_sums = p.sum(axis=1)
    summary = {
        "P_row_sums_close_to_1": bool(np.allclose(row_sums, 1.0, atol=1e-8)),
        "V_best_state": int(np.argmax(v)),
        "V_worst_state": int(np.argmin(v)),
        "V_best_value": float(v.max()),
        "V_worst_value": float(v.min()),
        "final_delta": float(conv[-1]),
    }
    save_json(results_dir / "data" / "stage04_summary.json", summary)
    return summary
