"""State containers shared by the Starling controller and dynamics."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DroneState:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class PendulumWorldState:
    drone: DroneState
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    t: float


@dataclass(frozen=True)
class BatchedDroneState:
    drone_position: np.ndarray
    drone_velocity: np.ndarray
    drone_rotation: np.ndarray
    drone_angular_velocity: np.ndarray

    @property
    def num_envs(self):
        return self.drone_position.shape[0]
