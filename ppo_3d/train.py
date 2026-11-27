"""Train PPO on the 3D drone double pendulum.

    uv run python train.py --steps 30_000_000 --envs 32 --name run1
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from dp.env import DronePendulumEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000_000)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--name", default="run1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-kwargs", default="{}", help='JSON, e.g. {"leash_radius": 0.5}')
    args = p.parse_args()

    torch.set_num_threads(4)
    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    import json
    env_kwargs = json.loads(args.env_kwargs)
    (out / "env_kwargs.json").write_text(json.dumps(env_kwargs))
    venv = make_vec_env(DronePendulumEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=4096,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh, log_std_init=-1.0),
        tensorboard_log=str(out / "tb"),
        device="cpu",
        seed=args.seed,
        verbose=1,
    )

    class SaveNorm(CheckpointCallback):
        def _on_step(self):
            ok = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                venv.save(str(out / "vecnormalize.pkl"))
            return ok

    cb = SaveNorm(save_freq=max(1, 2_000_000 // args.envs), save_path=str(out / "ckpt"), name_prefix="ppo")
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    model.save(str(out / "final"))
    venv.save(str(out / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()
