"""Training experiment orchestration using Stable-Baselines3 PPO."""

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecNormalize,
    VecVideoRecorder,
)

from drone_pendulum.training.config import PPOConfig
from drone_pendulum.training.run_utils import prepare_run_dir, write_run_card


class RLTrainer:
    """Own the lifecycle and artifacts for one PPO training run."""

    def __init__(
        self,
        config: PPOConfig,
        run_dir: Path,
        resume: Path | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.resume = resume
        self.train_env = None
        self.eval_env = None
        self.model = None

    def _make_env(self, seed: int, render_mode: str | None = None):
        def factory():
            env = gym.make(
                self.config.env_id,
                episode_limit=self.config.max_episode_len,
                render_mode=render_mode,
            )
            env.reset(seed=seed)
            return Monitor(env)

        return factory

    def _build_vec_env(self, seed: int):
        env = VecMonitor(DummyVecEnv([self._make_env(seed)]))
        if self.config.normalization_enabled:
            env = VecNormalize(
                env,
                norm_obs=self.config.normalize_observation,
                norm_reward=self.config.normalize_reward,
                clip_obs=10.0,
            )
        return env

    def _configure_evaluation_env(self) -> None:
        """Don't update norm stats during eval and report unnnormalized eval rewards."""
        if self.config.normalization_enabled:
            self.eval_env.training = False
            self.eval_env.norm_reward = False

    def _build_model(self):
        if self.resume is not None:
            return PPO.load(str(self.resume), env=self.train_env)
        return PPO(
            self.config.policy_type,
            self.train_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            tensorboard_log=str(self.run_dir / "tensorboard"),
            seed=self.config.seed,
            verbose=1,
        )

    def train(self) -> Path:
        """Train PPO and return the saved final model path."""
        prepare_run_dir(self.run_dir, self.config)
        self.train_env = self._build_vec_env(self.config.seed)
        self.eval_env = self._build_vec_env(self.config.seed + 1)
        self.model = self._build_model()

        self._configure_evaluation_env()

        callbacks = [
            CheckpointCallback(
                save_freq=self.config.checkpoint_freq,
                save_path=str(self.run_dir / "checkpoints"),
                name_prefix="ppo",
            ),
            EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.run_dir / "best_model"),
                log_path=str(self.run_dir),
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.eval_episodes,
                deterministic=True,
            ),
        ]

        try:
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
            )
            final_path = self.run_dir / "final_model"
            self.model.save(str(final_path))
            if self.config.normalization_enabled:
                self.train_env.save(str(self.run_dir / "vecnormalize.pkl"))
            write_run_card(self.run_dir, self.config, "complete")
            return final_path.with_suffix(".zip")
        finally:
            self.close()

    def evaluate(self, model_path: Path | None = None, episodes: int = 10) -> dict:
        """Evaluate a saved policy and return aggregate metrics."""
        model_path = model_path or self.run_dir / "final_model.zip"
        self.eval_env = self._build_vec_env(self.config.seed + 1)

        normalization_path = self.run_dir / "vecnormalize.pkl"
        if normalization_path.exists() and self.config.normalization_enabled:
            self.eval_env = VecNormalize.load(
                str(normalization_path), self.eval_env.venv
            )
        self._configure_evaluation_env()

        self.model = PPO.load(str(model_path), env=self.eval_env)
        returns = []
        lengths = []
        try:
            for _ in range(episodes):
                obs = self.eval_env.reset()
                total_reward = 0.0
                length = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done_array, _ = self.eval_env.step(action)
                    total_reward += float(np.asarray(reward).reshape(-1)[0])
                    length += 1
                    done = bool(done_array[0])
                returns.append(total_reward)
                lengths.append(length)
        finally:
            self.close()

        return {
            "episodes": episodes,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_episode_length": float(np.mean(lengths)),
        }

    def record_video(
        self,
        model_path: Path,
        video_dir: Path,
        label: str,
        episodes: int = 1,
    ) -> Path:
        """Record deterministic evaluation episodes for a saved policy."""
        video_dir.mkdir(parents=True, exist_ok=True)
        env = VecMonitor(
            DummyVecEnv([self._make_env(self.config.seed + 1, "rgb_array")])
        )
        normalization_path = self.run_dir / "vecnormalize.pkl"
        if normalization_path.exists() and self.config.normalization_enabled:
            env = VecNormalize.load(str(normalization_path), env.venv)
            env.training = False
            env.norm_reward = False
        env = VecVideoRecorder(
            env,
            str(video_dir),
            record_video_trigger=lambda step: True,
            video_length=self.config.max_episode_len * episodes,
            name_prefix=label,
        )
        model = PPO.load(str(model_path), env=env)
        try:
            for _ in range(episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done_array, _ = env.step(action)
                    done = bool(done_array[0])
        finally:
            env.close()
        return video_dir / f"{label}.mp4"

    def close(self) -> None:
        """Close any environments owned by the experiment."""
        if self.train_env is not None:
            self.train_env.close()
            self.train_env = None
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None
