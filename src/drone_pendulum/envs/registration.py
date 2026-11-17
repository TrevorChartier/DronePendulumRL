"""Gymnasium registration for the drone-pendulum environments."""

from pathlib import Path

from gymnasium.envs.registration import register, registry

from drone_pendulum.envs.drone_pendulum_env import DronePendulumEnv
from drone_pendulum.envs.starling_pendulum_env import StarlingPendulumEnv

ENV_ID = "DronePendulum-v0"
STARLING_ENV_ID = "StarlingPendulum-v0"
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "simulation"
    / "assets"
    / "drone_pendulum.xml"
)
STARLING_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "starling_simulation" / "assets" / "starling2max_pendulum.xml"
)


def make_drone_pendulum_env(
    model_path: str | None = None,
    episode_limit: int = 1000,
    **env_kwargs,
) -> DronePendulumEnv:
    """Create the registered environment with project defaults."""
    return DronePendulumEnv(
        model_path=str(MODEL_PATH if model_path is None else model_path),
        episode_limit=episode_limit,
        **env_kwargs,
    )


def register_envs() -> None:
    """Register project environments exactly once."""
    if ENV_ID not in registry:
        register(
            id=ENV_ID,
            entry_point="drone_pendulum.envs.registration:make_drone_pendulum_env",
        )
    if STARLING_ENV_ID not in registry:
        register(
            id=STARLING_ENV_ID,
            entry_point="drone_pendulum.envs.registration:make_starling_pendulum_env",
        )


def make_starling_pendulum_env(model_path=None, episode_limit=300, **env_kwargs):
    return StarlingPendulumEnv(
        model_path=str(STARLING_MODEL_PATH if model_path is None else model_path),
        episode_limit=episode_limit,
        **env_kwargs,
    )


register_envs()
