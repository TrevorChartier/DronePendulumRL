"""Maps normalized policy actions to forces/torques on the planar drone.

This mirrors simulation/actuation.py (the 3D version) so the action interface is the same
idea in 2D and 3D:

    action = [collective_thrust, roll_rate]   (both in [-1, 1])

  * collective_thrust: -1 -> 0 N, 0 -> hover thrust (total_mass * g), +1 -> 2x hover.
    So a zero action is (approximately) "hover", which is a nice property for RL and LQR.
  * roll_rate: desired body roll rate, scaled by max_body_rate. A simple proportional
    inner-loop controller turns it into a torque (this stands in for the Crazyflie's
    onboard rate controller).

Things you may want to revisit later (they are deliberately simple right now):
  * motor dynamics / first-order lag on thrust
  * actuation delay
  * a better inner rate controller (PI, feed-forward on inertia)
  * an alternative action space, e.g. two individual rotor thrusts [f_left, f_right]
"""

import numpy as np

from drone_pendulum.simulation.model_2d import PlanarDronePendulumModel

ACTION_DIM = 2


class PlanarActuation:
    """Converts normalized [thrust, roll_rate] actions into a world-frame wrench."""

    def __init__(
        self,
        drone_model: PlanarDronePendulumModel,
        max_body_rate: float = 5.0,
        rate_gain: float = 0.1,
        max_thrust_to_weight: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        drone_model : PlanarDronePendulumModel
        max_body_rate : float
            Roll rate (rad/s) commanded by action[1] = +/-1.
        rate_gain : float
            Proportional gain (N*m per rad/s) of the inner roll-rate loop.
        max_thrust_to_weight : float
            Thrust at action[0] = +1, as a multiple of total system weight.
        """
        self.drone_model = drone_model
        self.max_body_rate = max_body_rate
        self.rate_gain = rate_gain

        self.hover_thrust = drone_model.total_mass * drone_model.gravity  # N
        self.max_thrust = max_thrust_to_weight * self.hover_thrust  # N

    def action_to_thrust(self, action_thrust: float) -> float:
        """Map normalized thrust in [-1, 1] to Newtons in [0, max_thrust]."""
        return 0.5 * self.max_thrust * (action_thrust + 1.0)

    def thrust_to_action(self, thrust: float) -> float:
        """Inverse of action_to_thrust (useful for classical controllers)."""
        return 2.0 * thrust / self.max_thrust - 1.0

    def apply_action(self, action: np.ndarray) -> None:
        """Write the drone's thrust force and roll torque into data.xfrc_applied."""
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"Expected action shape ({ACTION_DIM},), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        data = self.drone_model.data
        drone_id = self.drone_model.drone_id

        thrust = self.action_to_thrust(action[0])

        desired_rate = action[1] * self.max_body_rate
        current_rate = self.drone_model.get_drone_roll_rate()
        roll_torque = self.rate_gain * (desired_rate - current_rate)

        # Thrust acts along the body z-axis; rotate it into the world frame.
        rotation = data.xmat[drone_id].reshape(3, 3)
        world_force = rotation @ np.array([0.0, 0.0, thrust])

        data.xfrc_applied[drone_id, :3] = world_force
        data.xfrc_applied[drone_id, 3:] = [roll_torque, 0.0, 0.0]  # torque about world X
