"""Manages the simulation

Has the following responsibilities:
    - Initialize and reset the simulation.
    - Accept control commands.
    - Advance physics by the appropriate number of steps.
    - Return the current physical state.
"""

import mujoco
import numpy as np

from drone_pendulum.simulation.model import DronePendulumModel
from drone_pendulum.simulation.actuation import Actuation


class DronePendulumSimulator:
    """Manages the MuJoCo simulation and drone actuation."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the simulation.

        Parameters
        ----------
        model_path : str
            Path to the MJCF model file.
        """
        self.drone_model = DronePendulumModel(model_path)
        self.actuation = Actuation(self.drone_model)

        self.model = self.drone_model.model
        self.data = self.drone_model.data

    def reset(
        self,
        joint_positions: np.ndarray | None = None,
        joint_velocities: np.ndarray | None = None,
    ) -> None:
        """Reset simulation and optionally set initial pendulum conditions."""

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        if joint_positions is not None or joint_velocities is not None:
            if joint_positions is None or joint_velocities is None:
                raise ValueError(
                    "Both joint_positions and joint_velocities must be provided."
                )

            self.drone_model.set_pendulum_state(
                joint_positions,
                joint_velocities,
            )

    def step(self, action: np.ndarray, n_substeps: int = 1) -> None:
        """
        Apply a control action and advance the physics simulation.

        Parameters
        ----------
        action : np.ndarray, shape (4,)
            Normalized PPO action.
        n_substeps : int, default=1
            Number of MuJoCo physics steps to execute.
        """
        if n_substeps < 1:
            raise ValueError("n_substeps must be at least 1.")

        for _ in range(n_substeps):
            self.actuation.apply_action(action)
            mujoco.mj_step(self.model, self.data)

    def get_state(self) -> dict[str, np.ndarray]:
        """
        Return the current physical state of the system.

        Returns
        -------
        dict[str, np.ndarray]
            Physical state quantities, including drone position,
            orientation, linear and angular velocity, and pendulum
            joint positions and velocities.
        """
        return {
            "drone_position": self.drone_model.get_drone_position(),
            "drone_orientation": self.drone_model.get_drone_orientation(),
            "drone_linear_velocity": self.drone_model.get_drone_linear_velocity(),
            "drone_angular_velocity": self.drone_model.get_drone_angular_velocity(
                frame="body"
            ),
            "joint_positions": self.drone_model.get_joint_positions(),
            "joint_velocities": self.drone_model.get_joint_velocities(),
        }
