"""Trajectory-tracking controller for the planar drone (payload-unaware).

Differential-flatness cascade, producing the planar env's action [thrust, roll_rate]:

    1. Position PD + acceleration feedforward:
           a_cmd = a_ref - Kp (p - p_ref) - Kd (v - v_ref)
    2. Desired force (gravity compensated with the TOTAL mass, drone + payload):
           F = m (a_cmd + g e_z) = [F_y, F_z]
    3. Thrust and attitude. Thrust acts along body z = (-sin(roll), cos(roll)), so
           roll_d = atan2(-F_y, F_z)            (clamped to +/- max_tilt)
           T      = F . b3                       (projection on the CURRENT body axis)
    4. Body-rate command = flatness feedforward + attitude P:
           roll_rate_ff = d/dt roll_d  with  dF/dt = m * jerk_ref
           roll_rate    = roll_rate_ff + k_att (roll_d - roll)
       The env's inner rate loop turns this into torque.

It reads only the drone's own state from the observation and ignores the pendulum,
which makes it the "payload-unaware cascaded controller" baseline from the proposal.
With a Hover() reference it is a plain position-hold controller.
"""

from __future__ import annotations

import numpy as np

from drone_pendulum.controllers.base import Controller
from drone_pendulum.controllers.trajectories import Hover, RefPoint, Trajectory


class PlanarTrackingController(Controller):
    """Tracks a Trajectory with the planar env (PlanarDronePendulumEnv or HoverEnv).

    The controller keeps its own clock: the k-th call to act() (after reset()) uses the
    reference at t = k * dt, which matches the sim time of the observation it receives.
    """

    def __init__(
        self,
        env,
        trajectory: Trajectory | None = None,
        kp=(4.0, 6.0),
        kd=(3.0, 4.0),
        k_att: float = 6.0,
        max_tilt: float = np.deg2rad(35.0),
        use_feedforward: bool = True,
    ):
        """
        env : planar env, used for constants (mass, gravity, action scaling, dt).
        kp, kd : (y, z) position / velocity gains in acceleration units (1/s^2, 1/s).
        k_att : roll attitude gain (1/s). Keep well below the inner rate-loop bandwidth.
        max_tilt : roll limit on the commanded attitude.
        use_feedforward : include acceleration + jerk feedforward (turn off to compare).
        """
        env = env.unwrapped
        model = env.sim.drone_model
        act = env.sim.actuation
        self.mass = model.total_mass
        self.g = model.gravity
        self.thrust_to_action = act.thrust_to_action
        self.max_thrust = act.max_thrust
        self.max_body_rate = act.max_body_rate
        self.dt = env.control_dt

        self.trajectory = trajectory or Hover()
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.k_att = float(k_att)
        self.max_tilt = float(max_tilt)
        self.use_feedforward = use_feedforward

        self.t = 0.0
        self.last_ref: RefPoint | None = None
        self.last_debug: dict = {}

    def reset(self) -> None:
        self.t = 0.0
        self.last_ref = None
        self.last_debug = {}

    def act(self, obs, info=None):
        obs = np.asarray(obs, dtype=float)
        # Planar obs layout (see planar_drone_pendulum_env.OBS_NAMES):
        # [y_err, z_err, sin_roll, cos_roll, sin_q1, cos_q1, sin_q2, cos_q2,
        #  y_dot, z_dot, roll_dot, q1_dot, q2_dot]
        pos = obs[0:2]  # relative to spawn
        roll = np.arctan2(obs[2], obs[3])
        vel = obs[8:10]

        ref = self.trajectory(self.t)
        self.last_ref = ref
        self.t += self.dt
        p_ref, v_ref = ref.pos[1:3], ref.vel[1:3]
        a_ref = ref.acc[1:3] if self.use_feedforward else np.zeros(2)
        j_ref = ref.jerk[1:3] if self.use_feedforward else np.zeros(2)

        # 1-2. desired force
        a_cmd = a_ref - self.kp * (pos - p_ref) - self.kd * (vel - v_ref)
        F = self.mass * (a_cmd + np.array([0.0, self.g]))
        F[1] = max(F[1], 0.1 * self.mass * self.g)  # never ask to push downwards

        # 3. attitude and thrust
        roll_d = np.arctan2(-F[0], F[1])
        tilt_clamped = abs(roll_d) > self.max_tilt
        roll_d = float(np.clip(roll_d, -self.max_tilt, self.max_tilt))
        b3 = np.array([-np.sin(roll), np.cos(roll)])
        thrust = float(np.clip(F @ b3, 0.0, self.max_thrust))

        # 4. body rate: flatness feedforward + attitude feedback
        F_dot = self.mass * j_ref
        rate_ff = 0.0 if tilt_clamped else (F[0] * F_dot[1] - F[1] * F_dot[0]) / float(F @ F)
        roll_rate = rate_ff + self.k_att * _wrap(roll_d - roll)

        self.last_debug = {"roll_d": roll_d, "thrust": thrust, "rate_ff": rate_ff,
                           "roll_rate_cmd": roll_rate}
        action = np.array([self.thrust_to_action(thrust), roll_rate / self.max_body_rate])
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
