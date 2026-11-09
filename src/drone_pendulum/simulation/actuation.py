
import numpy as np

from drone_pendulum.simulation.model import DronePendulumModel


class Actuation:
    """Converts normalized PPO actions into physical drone inputs."""

    def __init__(self, drone_model: DronePendulumModel, max_body_rate: float = 5.0,
                 rate_gain: float = 0.1) -> None:
        """
        Initialize the drone actuation controller.

        Parameters
        ----------
        drone_model : DronePendulumModel
            Loaded MuJoCo drone-pendulum model.
        max_body_rate : float, default=5.0
            Maximum commanded body angular velocity in rad/s.
        rate_gain : float, default=0.1
            Proportional body-rate controller gain.
        """
        self.drone_model = drone_model
        self.max_body_rate = max_body_rate
        self.rate_gain = rate_gain

        self.total_mass = drone_model.total_mass

        # Maximum collective thrust (N).
        self.max_thrust = 2.0 * self.total_mass * 9.81

    def apply_action(self, action: np.ndarray) -> None:
        """
        Take normalized PPO actions and apply them to the drone model.

        Parameters
        ----------
        action : np.ndarray, shape (4,)
            Normalized actions in [-1, 1]:
            [collective_thrust, roll_rate, pitch_rate, yaw_rate].
        """
        if np.shape(action) != (4,):
            raise ValueError(f"Expected action shape (4,), got {np.shape(action)}")

        action = np.clip(action, -1.0, 1.0)

        data = self.drone_model.data
        drone_id = self.drone_model.drone_id

        thrust = 0.5 * self.max_thrust * (action[0] + 1.0) # map thrust from [-1,1] to [0, max_thrust]

        desired_angular_vel = action[1:4] * self.max_body_rate
        current_angular_vel = self.drone_model.get_drone_angular_velocity(frame="body")
        body_torque = self.rate_gain * (desired_angular_vel - current_angular_vel)

        # Rotate thrust and torque into world-frame coordinates.
        curr_drone_rotation = data.xmat[drone_id].reshape(3, 3)
        world_force = curr_drone_rotation @ np.array([0.0, 0.0, thrust])
        world_torque = curr_drone_rotation @ body_torque

        # Apply external force and torque to the drone.
        data.xfrc_applied[drone_id, :3] = world_force
        data.xfrc_applied[drone_id, 3:] = world_torque
