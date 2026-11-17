"""Drone-pendulum reinforcement-learning package."""

# Importing the package registers its Gymnasium environments.
from drone_pendulum.envs.registration import register_envs

__all__ = ["register_envs"]
