from __future__ import annotations

from pathlib import Path

import numpy as np
import pygame

from src.core.env import env_from_jsonable
from src.core.sim import run_episode
from src.io.artifacts import load_json, load_npy, save_csv, save_json, stage_figure_dir
from src.render.pygame_renderer import draw_grid, init_pygame, save_surface


def _compose_2x2(images: list[Path], out_path: Path, title: str) -> None:
    init_pygame()
    loaded = [pygame.image.load(str(p)) for p in images]
    w, h = loaded[0].get_width(), loaded[0].get_height()
    canvas = pygame.Surface((w * 2, h * 2 + 60))
    canvas.fill((255, 255, 255))
    font = pygame.font.SysFont("Arial", 26, bold=True)
    canvas.blit(font.render(title, True, (20, 20, 20)), (20, 15))
    positions = [(0, 60), (w, 60), (0, 60 + h), (w, 60 + h)]
    for img, pos in zip(loaded, positions):
        canvas.blit(img, pos)
    save_surface(canvas, out_path)


def run(results_dir: Path) -> dict:
    env_payload = load_json(results_dir / "data" / "env.json")
    env = env_from_jsonable(env_payload)
    pi_random = load_npy(results_dir / "data" / "policy_init.npy")
    pi_improved = load_npy(results_dir / "data" / "policy_improved.npy")

    free_starts = [s for s in range(env.num_states) if s != env.goal_state and s not in env.obstacle_set]
    rng = np.random.default_rng(int(env_payload["seed"]) + 123)
    starts = rng.choice(free_starts, size=2, replace=False).tolist()

    rows = []
    fig_dir = stage_figure_dir(results_dir, "07_compare")
    board_images: list[Path] = []

    for p_name, pi in [("random", pi_random), ("improved", pi_improved)]:
        for start in starts:
            sh, rh = run_episode(env, int(start), pi, seed=int(env_payload["seed"]), max_steps=5000)
            rows.append(
                {
                    "policy": p_name,
                    "start_state": int(start),
                    "steps": int(len(sh) - 1),
                    "total_reward": float(sum(rh)),
                    "reached_goal": int(sh[-1] == env.goal_state),
                }
            )
            out = fig_dir / f"traj_{p_name}_start_{int(start)}.png"
            draw_grid(
                env,
                out,
                f"{p_name.title()} policy trajectory (start={int(start)})",
                visited_states=set(sh),
                trajectory=sh,
                start_state=int(start),
            )
            board_images.append(out)

    save_csv(results_dir / "tables" / "final_comparison.csv", rows)
    _compose_2x2(board_images, fig_dir / "comparison_board.png", "Random vs Improved Policy")

    random_steps = [r["steps"] for r in rows if r["policy"] == "random"]
    improved_steps = [r["steps"] for r in rows if r["policy"] == "improved"]

    summary = {
        "starts": starts,
        "avg_steps_random": float(np.mean(random_steps)),
        "avg_steps_improved": float(np.mean(improved_steps)),
        "improved_not_worse": bool(np.mean(improved_steps) <= np.mean(random_steps)),
    }
    save_json(results_dir / "data" / "stage07_summary.json", summary)
    return summary
