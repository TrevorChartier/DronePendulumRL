"""Command-line entry point for PPO evaluation."""

import argparse
import json
from pathlib import Path

from drone_pendulum.training import PPOConfig, RLTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run_dir
    config = PPOConfig(**json.loads((run_dir / "config.json").read_text()))
    model = args.model or run_dir / "final_model.zip"
    experiment = RLTrainer(config, run_dir)
    print(json.dumps(experiment.evaluate(model, args.episodes), indent=2))


if __name__ == "__main__":
    main()
