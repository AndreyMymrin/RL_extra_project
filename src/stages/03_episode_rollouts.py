from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.env import env_from_jsonable
from src.core.sim import run_episode
from src.io.artifacts import load_json, load_npy, save_csv, save_json, stage_figure_dir
from src.render.pygame_renderer import draw_grid


def run(results_dir: Path) -> dict:
    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    policy = load_npy(results_dir / "data" / "policy_init.npy")

    free_starts = [s for s in range(env.num_states) if s != env.goal_state and s not in env.obstacle_set]

    rows = []
    for start in free_starts:
        sh, rh = run_episode(env, start, policy, seed=int(env_payload["seed"]), max_steps=5000)
        rows.append(
            {
                "start_state": int(start),
                "steps": int(len(sh) - 1),
                "total_reward": float(sum(rh)),
                "reached_goal": int(sh[-1] == env.goal_state),
            }
        )

    save_csv(results_dir / "tables" / "episodes_random_policy.csv", rows)

    # save two representative trajectory plots
    fig_dir = stage_figure_dir(results_dir, "03_episodes")
    rng = np.random.default_rng(int(env_payload["seed"]))
    demo_starts = rng.choice(free_starts, size=2, replace=False)
    for idx, start in enumerate(demo_starts):
        sh, _ = run_episode(env, int(start), policy, seed=int(env_payload["seed"]), max_steps=5000)
        draw_grid(
            env,
            fig_dir / f"trajectory_random_{idx+1}.png",
            f"Trajectory under initial policy (start={int(start)})",
            visited_states=set(sh),
            trajectory=sh,
            start_state=int(start),
        )

    summary = {
        "num_episodes": len(rows),
        "all_reached_goal": bool(all(r["reached_goal"] == 1 for r in rows)),
        "avg_steps": float(np.mean([r["steps"] for r in rows])),
    }
    save_json(results_dir / "data" / "stage03_summary.json", summary)
    return summary
