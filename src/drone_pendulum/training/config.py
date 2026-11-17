"""Configuration for reproducible PPO experiments."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PPOConfig:
    env_id: str = "DronePendulum-v0"
    max_episode_len: int = 1000
    seed: int = 0
    total_timesteps: int = 100_000
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_observation: bool = True
    normalize_reward: bool = False
    checkpoint_freq: int = 25_000
    eval_freq: int = 10_000
    eval_episodes: int = 10

    @property
    def normalization_enabled(self) -> bool:
        """Whether the environment should use VecNormalize."""
        return self.normalize_observation or self.normalize_reward

    def as_dict(self) -> dict:
        return asdict(self)
