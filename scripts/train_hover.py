# This trains PPO on hover using SB3 defaults and saves the policy with its observation stats.

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_pendulum.envs.hover_env import HoverConfig, HoverEnv


class RewardLogger(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            for name, value in info.get("reward_terms", {}).items():
                self.logger.record_mean(f"reward/{name}", value)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.timesteps <= 0 or args.num_envs <= 0:
        parser.error("--timesteps and --num-envs must be positive")
    run_dir = args.run_dir or Path("runs") / datetime.now().strftime("hover_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = HoverConfig()
    env = VecNormalize(DummyVecEnv([lambda: Monitor(HoverEnv(cfg)) for _ in range(args.num_envs)]),
                       norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1,
                tensorboard_log=str(run_dir / "tensorboard"))
    metadata = {"environment": asdict(cfg), "seed": args.seed,
                "requested_timesteps": args.timesteps, "num_envs": args.num_envs,
                "ppo": "SB3 defaults",
                "normalization": "observations only"}
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    callback = CheckpointCallback(max(25_000 // args.num_envs, 1), str(run_dir / "checkpoints"),
                                  name_prefix="ppo", save_vecnormalize=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=[callback, RewardLogger()])
        model.save(run_dir / "policy")
        env.save(str(run_dir / "vecnormalize.pkl"))
    finally:
        env.close()
    print(f"Saved policy and normalization to {run_dir}")


if __name__ == "__main__":
    main()
