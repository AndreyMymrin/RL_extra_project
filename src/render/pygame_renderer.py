from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pygame

from src.core.env import EnvState, state_to_rc


_initialized = False


def init_pygame() -> None:
    global _initialized
    if _initialized:
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    _initialized = True


def save_surface(surface: pygame.Surface, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(out_path))


def _lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _value_color(x: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if vmax - vmin < 1e-12:
        t = 0.5
    else:
        t = (x - vmin) / (vmax - vmin)
    # red -> yellow -> green
    if t < 0.5:
        return _lerp_color((200, 50, 50), (240, 220, 80), t * 2)
    return _lerp_color((240, 220, 80), (40, 160, 80), (t - 0.5) * 2)


def draw_grid(
    env: EnvState,
    out_path: Path,
    title: str,
    cell_values: np.ndarray | None = None,
    visited_states: set[int] | None = None,
    trajectory: list[int] | None = None,
    start_state: int | None = None,
    best_actions: np.ndarray | None = None,
) -> None:
    init_pygame()

    cell = 70
    margin = 60
    width = env.grid_w * cell + margin * 2
    height = env.grid_h * cell + margin * 2 + 60
    surf = pygame.Surface((width, height))
    surf.fill((250, 250, 250))

    font_title = pygame.font.SysFont("Arial", 28, bold=True)
    font_cell = pygame.font.SysFont("Arial", 15)
    font_small = pygame.font.SysFont("Arial", 13)
    title_s = font_title.render(title, True, (20, 20, 20))
    surf.blit(title_s, (margin, 12))

    vmin = float(np.min(cell_values)) if cell_values is not None else -1.0
    vmax = float(np.max(cell_values)) if cell_values is not None else 1.0

    for s in range(env.num_states):
        r, c = state_to_rc(s, env.grid_w)
        rp = env.grid_h - 1 - r
        x = margin + c * cell
        y = margin + 40 + rp * cell

        if s == env.goal_state:
            color = (46, 204, 113)
        elif s in env.obstacle_set:
            color = (77, 77, 77)
        elif visited_states and s in visited_states:
            color = (174, 214, 241)
        elif cell_values is not None:
            color = _value_color(float(cell_values[s]), vmin, vmax)
        else:
            color = (214, 234, 248)

        pygame.draw.rect(surf, color, (x, y, cell, cell))
        pygame.draw.rect(surf, (20, 20, 20), (x, y, cell, cell), 2)

        label = f"s={s}"
        if s == env.goal_state:
            label += " G"
        if s in env.obstacle_set:
            label += " X"

        txt = font_cell.render(label, True, (255, 255, 255) if s in env.obstacle_set else (0, 0, 0))
        surf.blit(txt, (x + 6, y + 6))

        if cell_values is not None:
            vtxt = font_small.render(f"V={cell_values[s]:.1f}", True, (0, 0, 0))
            surf.blit(vtxt, (x + 6, y + cell - 20))

        if best_actions is not None and s not in env.obstacle_set and s != env.goal_state:
            arrow = ["U", "D", "L", "R"][int(best_actions[s])]
            atxt = font_cell.render(arrow, True, (20, 40, 120))
            surf.blit(atxt, (x + cell - 20, y + cell - 22))

        if start_state is not None and s == start_state:
            pygame.draw.circle(surf, (243, 156, 18), (x + cell // 2, y + cell // 2), 8)

    if trajectory and len(trajectory) > 1:
        points = []
        for s in trajectory:
            r, c = state_to_rc(s, env.grid_w)
            rp = env.grid_h - 1 - r
            points.append((margin + c * cell + cell // 2, margin + 40 + rp * cell + cell // 2))
        for p0, p1 in zip(points[:-1], points[1:]):
            pygame.draw.line(surf, (200, 30, 30), p0, p1, 3)

    save_surface(surf, out_path)


def draw_matrix_heatmap(
    matrix: np.ndarray,
    out_path: Path,
    title: str,
    max_cells: int = 40,
) -> None:
    init_pygame()

    rows, cols = matrix.shape
    display_rows = min(rows, max_cells)
    display_cols = min(cols, max_cells)
    m = matrix[:display_rows, :display_cols]

    cell = 20
    margin = 50
    width = display_cols * cell + margin * 2
    height = display_rows * cell + margin * 2 + 50

    surf = pygame.Surface((width, height))
    surf.fill((255, 255, 255))
    font_title = pygame.font.SysFont("Arial", 24, bold=True)
    title_s = font_title.render(title, True, (20, 20, 20))
    surf.blit(title_s, (margin, 12))

    vmin = float(np.min(m))
    vmax = float(np.max(m))
    for r in range(display_rows):
        for c in range(display_cols):
            x = margin + c * cell
            y = margin + 40 + r * cell
            color = _value_color(float(m[r, c]), vmin, vmax)
            pygame.draw.rect(surf, color, (x, y, cell, cell))
            pygame.draw.rect(surf, (30, 30, 30), (x, y, cell, cell), 1)

    save_surface(surf, out_path)


def draw_line_plot(
    series: list[np.ndarray],
    labels: list[str],
    out_path: Path,
    title: str,
    y_label: str,
    log_y: bool = False,
) -> None:
    init_pygame()

    width, height = 1100, 500
    margin_l, margin_r, margin_t, margin_b = 70, 20, 60, 60
    surf = pygame.Surface((width, height))
    surf.fill((255, 255, 255))

    font_title = pygame.font.SysFont("Arial", 24, bold=True)
    font = pygame.font.SysFont("Arial", 16)
    surf.blit(font_title.render(title, True, (20, 20, 20)), (margin_l, 15))
    surf.blit(font.render(y_label, True, (20, 20, 20)), (10, margin_t))

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    pygame.draw.rect(surf, (0, 0, 0), (margin_l, margin_t, plot_w, plot_h), 2)

    max_len = max(len(s) for s in series)
    all_vals = np.concatenate([np.asarray(s, dtype=float) for s in series])
    if log_y:
        all_vals = np.log10(np.maximum(all_vals, 1e-12))
    ymin, ymax = float(np.min(all_vals)), float(np.max(all_vals))
    if abs(ymax - ymin) < 1e-12:
        ymax += 1.0

    colors = [(52, 152, 219), (231, 76, 60), (46, 204, 113), (155, 89, 182)]
    for idx, s in enumerate(series):
        yvals = np.asarray(s, dtype=float)
        if log_y:
            yvals = np.log10(np.maximum(yvals, 1e-12))
        pts = []
        for i, y in enumerate(yvals):
            x = margin_l + int((i / max(1, max_len - 1)) * plot_w)
            yy = margin_t + plot_h - int(((y - ymin) / (ymax - ymin)) * plot_h)
            pts.append((x, yy))
        if len(pts) > 1:
            pygame.draw.lines(surf, colors[idx % len(colors)], False, pts, 2)
        if pts:
            lx, ly = pts[-1]
            surf.blit(font.render(labels[idx], True, colors[idx % len(colors)]), (min(lx + 8, width - 120), ly))

    save_surface(surf, out_path)


def draw_bar_chart(values: np.ndarray, out_path: Path, title: str, obstacle_set: set[int]) -> None:
    init_pygame()
    width, height = 1500, 500
    margin_l, margin_r, margin_t, margin_b = 60, 20, 60, 60
    surf = pygame.Surface((width, height))
    surf.fill((255, 255, 255))

    font_title = pygame.font.SysFont("Arial", 24, bold=True)
    surf.blit(font_title.render(title, True, (20, 20, 20)), (margin_l, 15))

    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    pygame.draw.rect(surf, (0, 0, 0), (margin_l, margin_t, plot_w, plot_h), 2)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-12:
        vmax += 1.0

    n = len(values)
    bw = max(2, plot_w // max(1, n))

    for i, v in enumerate(values):
        x = margin_l + i * bw
        h = int(((float(v) - vmin) / (vmax - vmin)) * plot_h)
        y = margin_t + plot_h - h
        if i in obstacle_set:
            color = (77, 77, 77)
        else:
            color = _value_color(float(v), vmin, vmax)
        pygame.draw.rect(surf, color, (x, y, bw - 1, h))

    save_surface(surf, out_path)
