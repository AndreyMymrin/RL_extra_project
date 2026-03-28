from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dirs(results_dir: Path) -> dict[str, Path]:
    data_dir = results_dir / "data"
    fig_dir = results_dir / "figures"
    table_dir = results_dir / "tables"
    for p in (results_dir, data_dir, fig_dir, table_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {"results": results_dir, "data": data_dir, "figures": fig_dir, "tables": table_dir}


def stage_figure_dir(results_dir: Path, stage_name: str) -> Path:
    p = results_dir / "figures" / stage_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_npy(path: Path) -> np.ndarray:
    return np.load(path)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def assert_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty artifact: {path}")
