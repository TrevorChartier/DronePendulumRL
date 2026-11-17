"""Payload-UNAWARE cascaded controller (secondary baseline in the proposal).

A standard planar quadrotor position controller that ignores the pendulum entirely
(treats it as a disturbance). Comparing against this tells you how much explicitly
reasoning about the payload actually buys you.

Cascade (outer -> inner):
    position error (y, z)  --PD-->  desired accelerations (a_y, a_z)
    (a_y, a_z)             ------>  collective thrust T and desired roll angle
    roll error             --P--->  desired roll rate  (the env's inner loop handles the rest)

Planar quadrotor relations (thrust along body z, roll about +X):
    m * a_y = -T * sin(roll)
    m * a_z =  T * cos(roll) - m * g
=> T = m * sqrt(a_y^2 + (a_z + g)^2),  roll_des = atan2(-a_y, a_z + g)

Double-check the sign of a_y vs roll in the sim before trusting the formulas above!
"""

import numpy as np

from drone_pendulum.controllers.base import Controller


class CascadedPDController(Controller):
    def __init__(self, env, kp_pos=(2.0, 4.0), kd_pos=(2.0, 3.0), kp_roll=5.0):
        """
        env is used only to read physical constants and action scaling.
        Gains are placeholders; tuning them is part of the exercise.
        """
        act = env.sim.actuation
        self.mass = env.sim.drone_model.total_mass
        self.g = env.sim.drone_model.gravity
        self.thrust_to_action = act.thrust_to_action
        self.max_body_rate = act.max_body_rate

        self.kp_pos = np.asarray(kp_pos)
        self.kd_pos = np.asarray(kd_pos)
        self.kp_roll = kp_roll

    def act(self, obs, info):
        x = info["state"]
        # x = [y, z, roll, q1, q2, y_dot, z_dot, roll_dot, q1_dot, q2_dot]
        # Note: the pendulum entries are deliberately ignored here.

        # obs[0:2] is the position error (pos - target) in world (y, z).
        # TODO(you):
        #   1. a_des = -kp_pos * pos_err - kd_pos * vel
        #   2. thrust T and roll_des from the relations in the module docstring
        #      (consider clamping roll_des to e.g. +/- 30 deg)
        #   3. roll_rate_des = kp_roll * (roll_des - roll)
        #   4. normalize: [self.thrust_to_action(T), roll_rate_des / self.max_body_rate]
        a_des = -self.kp_pos * obs[0:2] - self.kd_pos * x[5:7]
        T = self.mass * np.sqrt(a_des[0]**2 + (a_des[1] + self.g)**2)
        roll_des = np.arctan2(-a_des[0], a_des[1] + self.g)
        roll_rate_des = self.kp_roll * (roll_des - x[2])
        return np.array([self.thrust_to_action(T), roll_rate_des / self.max_body_rate])
