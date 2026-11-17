"""Gymnasium registration for the drone-pendulum environments."""

from pathlib import Path

from gymnasium.envs.registration import register, registry

from drone_pendulum.envs.drone_pendulum_env import DronePendulumEnv

ENV_ID = "DronePendulum-v0"
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "simulation"
    / "assets"
    / "drone_pendulum.xml"
)


def make_drone_pendulum_env(
    model_path: str | None = None,
    max_episode_len: int = 1000,
    **env_kwargs,
) -> DronePendulumEnv:
    """Create the registered environment with project defaults."""
    return DronePendulumEnv(
        model_path=str(MODEL_PATH if model_path is None else model_path),
        max_episode_len=max_episode_len,
        **env_kwargs,
    )


def register_envs() -> None:
    """Register project environments exactly once."""
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="drone_pendulum.envs.registration:make_drone_pendulum_env",
        )


register_envs()
