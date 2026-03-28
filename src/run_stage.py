from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

from src.io.artifacts import ensure_dirs


_STAGE_FILES = {
    "env": "01_environment.py",
    "dynamics": "02_dynamics_and_sampling.py",
    "episodes": "03_episode_rollouts.py",
    "value": "04_value_iteration.py",
    "q": "05_q_iteration.py",
    "improve": "06_policy_improvement.py",
    "compare": "07_final_comparison.py",
}


def _load_stage_module(stage_key: str) -> ModuleType:
    file_name = _STAGE_FILES[stage_key]
    path = Path(__file__).resolve().parent / "stages" / file_name
    mod_name = f"stage_{stage_key}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load stage module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(stage: str, results_dir: Path, seed: int | None) -> dict:
    module = _load_stage_module(stage)
    if stage == "env":
        return module.run(results_dir=results_dir, seed=seed)
    return module.run(results_dir=results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL+ stage pipeline")
    parser.add_argument("--stage", required=True, choices=[*list(_STAGE_FILES.keys()), "all"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ensure_dirs(results_dir)

    stages = list(_STAGE_FILES.keys()) if args.stage == "all" else [args.stage]

    for stage in stages:
        summary = run_one(stage, results_dir=results_dir, seed=args.seed)
        print(f"[stage:{stage}] {summary}")


if __name__ == "__main__":
    main()
