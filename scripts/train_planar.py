"""Training entry point for the planar env. TODO(you): this is a skeleton.

Two reasonable routes (the proposal compares PPO, SAC and TD3):

  A) Library: Stable-Baselines3 (pip install stable-baselines3; add it to pyproject.toml)
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        vec_env = make_vec_env(make_env, n_envs=8)
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="runs/")
        model.learn(total_timesteps=2_000_000)
        model.save("checkpoints/ppo_planar")
     Fast to get going; lets you focus on reward/obs/DR design.

  B) Your own PPO/SAC in drone_pendulum/rl/ (closer to the course homework). More
     learning, more debugging.

Either way, before a long run:
  * Implement the env's reward and termination (PlanarDronePendulumEnv TODOs).
  * Get the LQR baseline working first: if LQR can't stabilize small perturbations,
    something is off in the sim/actuation, and RL won't fix it.
  * Normalize observations (e.g. SB3's VecNormalize) and log episode return AND your
    physical metrics (evaluation/metrics.py), not just return.
  * Fix seeds and save the config alongside each checkpoint.
"""

import argparse

import gymnasium as gym

import drone_pendulum.envs  # noqa: F401  (registers PlanarDronePendulum-v0)
from drone_pendulum.envs.planar_drone_pendulum_env import PlanarEnvConfig


def make_env(seed: int = 0):
    cfg = PlanarEnvConfig()  # TODO(you): load from configs/ once you have one
    env = gym.make("PlanarDronePendulum-v0", config=cfg)
    env.reset(seed=seed)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="ppo", choices=["ppo", "sac", "td3"])
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = make_env(args.seed)
    print(f"obs space: {env.observation_space}\naction space: {env.action_space}")

    # TODO(you): build the agent for args.algo, train, checkpoint, evaluate.
    raise NotImplementedError("Training loop not implemented yet. See module docstring.")


if __name__ == "__main__":
    main()
