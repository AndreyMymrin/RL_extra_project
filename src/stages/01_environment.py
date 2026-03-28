from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.env import EnvConfig, env_to_jsonable, make_environment
from src.io.artifacts import ensure_dirs, save_csv, save_json, save_npy, stage_figure_dir
from src.render.pygame_renderer import draw_grid


def run(results_dir: Path, seed: int | None = None) -> dict:
    dirs = ensure_dirs(results_dir)
    cfg = EnvConfig(seed=42 if seed is None else int(seed))
    env, policy = make_environment(cfg)

    save_json(dirs["data"] / "env.json", env_to_jsonable(env, cfg.seed, cfg.gamma))
    save_npy(dirs["data"] / "policy_init.npy", policy)

    rows = []
    for s in range(env.num_states):
        r, c = divmod(s, env.grid_w)
        rows.append(
            {
                "state": s,
                "row": r,
                "col": c,
                "is_goal": int(s == env.goal_state),
                "is_obstacle": int(s in env.obstacle_set),
                "reward": float(env.rewards[s]),
            }
        )
    save_csv(dirs["tables"] / "environment_states.csv", rows)

    fig_dir = stage_figure_dir(results_dir, "01_environment")
    draw_grid(env, fig_dir / "grid_map.png", "Grid World: Goal and Obstacles")

    summary = {
        "seed": cfg.seed,
        "grid_h": env.grid_h,
        "grid_w": env.grid_w,
        "num_states": env.num_states,
        "goal_state": env.goal_state,
        "obstacle_count": len(env.obstacle_states),
        "obstacle_states": [int(x) for x in env.obstacle_states.tolist()],
        "policy_row_sums_ok": bool(np.allclose(policy.sum(axis=1), 1.0)),
    }
    save_json(dirs["data"] / "stage01_summary.json", summary)
    return summary
