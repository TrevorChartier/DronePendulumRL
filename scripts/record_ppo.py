"""Record videos for selected PPO checkpoints."""

import argparse
import json
from pathlib import Path

from drone_pendulum.training import PPOConfig, TrainingExperiment


def checkpoint_for_iteration(run_dir: Path, iteration: int) -> Path:
    path = run_dir / "checkpoints" / f"ppo_{iteration}_steps.zip"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("iterations", nargs="+", type=int)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = PPOConfig(**json.loads((args.run_dir / "config.json").read_text()))
    output_dir = args.output_dir or args.run_dir / "videos"
    experiment = TrainingExperiment(config, args.run_dir)
    for iteration in args.iterations:
        checkpoint = checkpoint_for_iteration(args.run_dir, iteration)
        label = f"checkpoint_{iteration}"
        path = experiment.record_video(
            checkpoint,
            output_dir,
            label,
            episodes=args.episodes,
        )
        print(path)


if __name__ == "__main__":
    main()
