"""Gymnasium registration for the drone-pendulum environments."""

from pathlib import Path

from gymnasium.envs.registration import register, registry

from drone_pendulum.envs.drone_pendulum_env import DronePendulumEnv
from drone_pendulum.envs.starling_pendulum_env import StarlingPendulumEnv

ENV_ID = "DronePendulum-v0"
STARLING_ENV_ID = "StarlingPendulum-v0"
PLANAR_ENV_ID = "PlanarDronePendulum-v0"
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "simulation"
    / "assets"
    / "drone_pendulum.xml"
)
# Models built from the hardware CAD (0.6 kg airframe, 100 mm + 75 mm links). Same
# body/joint names as the defaults, so pass them as model_path to the existing envs:
#   gym.make(ENV_ID, model_path=str(CAD_MODEL_PATH))
#   PlanarDronePendulumEnv(PlanarEnvConfig(model_path=str(PLANAR_CAD_MODEL_PATH)))
CAD_MODEL_PATH = MODEL_PATH.with_name("drone_pendulum_cad.xml")
PLANAR_CAD_MODEL_PATH = MODEL_PATH.with_name("drone_pendulum_2d_cad.xml")
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
    if PLANAR_ENV_ID not in registry:
        register(
            id=PLANAR_ENV_ID,
            entry_point="drone_pendulum.envs.planar_drone_pendulum_env:PlanarDronePendulumEnv",
        )


def make_starling_pendulum_env(model_path=None, episode_limit=300, **env_kwargs):
    return StarlingPendulumEnv(
        model_path=str(STARLING_MODEL_PATH if model_path is None else model_path),
        episode_limit=episode_limit,
        **env_kwargs,
    )


register_envs()
