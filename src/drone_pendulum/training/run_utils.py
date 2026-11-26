"""Small helpers for experiment directories and run documentation."""

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch


def _config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "as_dict") and callable(config.as_dict):
        return config.as_dict()
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError("config must be a mapping, dataclass instance, or expose .as_dict()")


def git_commit() -> str | None:
    """Return the current Git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_run_dir(run_dir: Path, config: Any) -> None:
    """Create a new run directory and write its initial configuration."""
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "tensorboard"):
        (run_dir / name).mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(_config_to_dict(config), indent=2) + "\n"
    )
    write_run_card(run_dir, config, "running")


def write_run_card(
    run_dir: Path,
    config: Any,
    status: str,
    artifacts: dict[str, str] | None = None,
) -> None:
    """Write human-readable metadata describing an experiment run."""
    card = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "config": _config_to_dict(config),
        "artifacts": artifacts
        or {
            "checkpoints": "checkpoints/",
            "tensorboard": "tensorboard/",
        },
    }
    (run_dir / "run_card.json").write_text(json.dumps(card, indent=2) + "\n")
