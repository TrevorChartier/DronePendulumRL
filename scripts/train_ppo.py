"""Command-line entry point for PPO training."""

import argparse
from datetime import datetime
from pathlib import Path

from drone_pendulum.training import PPOConfig, RLTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--max-episode-len", type=int, default=1000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    name = args.exp_name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.runs_dir / name
    config = PPOConfig(
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        max_episode_len=args.max_episode_len,
    )
    RLTrainer(config, run_dir, resume=args.resume).train()


if __name__ == "__main__":
    main()
