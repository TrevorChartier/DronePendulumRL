"""Small helpers for experiment directories and run documentation."""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from drone_pendulum.training.config import PPOConfig


def git_commit() -> str | None:
    """Return the current Git commit when available.
    
    Useful for documenting exactly  what src code was used to train
    model.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_run_dir(run_dir: Path, config: PPOConfig) -> None:
    """Create a new run directory and write its initial configuration."""
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "best_model", "tensorboard"):
        (run_dir / name).mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2) + "\n"
    )
    write_run_card(run_dir, config, "running")


def write_run_card(run_dir: Path, config: PPOConfig, status: str) -> None:
    """Write human-readable metadata describing an experiment run."""
    card = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": config.as_dict(),
        "artifacts": {
            "final_model": "final_model.zip",
            "normalization": "vecnormalize.pkl",
            "checkpoints": "checkpoints/",
            "best_model": "best_model/",
            "tensorboard": "tensorboard/",
        },
    }
    (run_dir / "run_card.json").write_text(json.dumps(card, indent=2) + "\n")
