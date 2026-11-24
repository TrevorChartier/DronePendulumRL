"""Geometric (SE(3)) trajectory-tracking controller for the 3D box-model drone.

Produces the 3D env's action [collective_thrust, roll_rate, pitch_rate, yaw_rate]
(normalized, body-frame rates) for DronePendulumEnv. It is the rate-level version of
Lee et al. / Mellinger & Kumar:

    1. Position PD + acceleration feedforward, gravity compensated with TOTAL mass:
           F = m (a_ref - Kp e_p - Kd e_v + g e3)       (horizontal part tilt-limited)
    2. Thrust along the current body z:      T = F . b3
    3. Desired attitude from F and yaw:      b3_d = F/|F|,  b1_d ~ [cos yaw, sin yaw, 0]
    4. Attitude error (body frame):          e_R = 1/2 vee(R_d^T R - R^T R_d)
    5. Desired body rates = flatness feedforward - attitude P:
           h        = m/|F| (j_ref - (b3_d . j_ref) b3_d)
           w_d      = [-h . b2_d,  h . b1_d,  yaw_rate (e3 . b3_d)]   (desired body frame)
           w_des    = R^T R_d w_d - k_att * e_R
    6. Rate-loop compensation (inertia aware), see below:
           tau      = J_eff k_rate (w_des - w) + w x J_eff w
           w_cmd    = w + tau / rate_loop_gain

Why step 6: the pendulum hangs from hinges (bearings) that only rotate about the drone's
body x-axis, matching the real hardware. About x the payload swings freely, but about
y it is rigidly attached, so the drone has to rotate the whole 1 m payload when it
pitches: the effective pitch inertia is ~11x the roll inertia. The env's inner rate
loop is a plain proportional loop, torque = rate_loop_gain * (w_cmd - w), with the same
gain on every axis, so a raw rate command is tracked ~11x more slowly in pitch than in
roll and a controller that just sends w_des goes unstable as soon as it needs to pitch.
Step 6 asks for the torque that would give the desired rate response (bandwidth k_rate)
given the real effective inertia, and converts it back into the rate command that makes
the P loop produce that torque. J_eff is computed from the MuJoCo model, so it follows
the XML (including which axis the hinges are on).

On hardware the inner loop is the flight controller's own rate controller, not this P
loop: set compensate_rate_loop=False if that loop already tracks rates well with the
payload attached, or set rate_loop_gain / k_rate to match it.

Like the planar controller it uses only the drone's own state (payload-unaware).

The Starling simulator has its own thrust-level SE3Controller
(starling_simulation/controller.py); this one targets the box model's thrust + body
rate action space, which is the interface the RL policy uses.
"""

from __future__ import annotations

import mujoco
import numpy as np

from drone_pendulum.controllers.base import Controller
from drone_pendulum.controllers.trajectories import Hover, RefPoint, Trajectory

E3 = np.array([0.0, 0.0, 1.0])


def quat_to_rotmat(q_wxyz) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q_wxyz, dtype=float))
    return R.reshape(3, 3)


def _vee(W: np.ndarray) -> np.ndarray:
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def effective_rotational_inertia(model: mujoco.MjModel, free_joint: str = "drone_free") -> np.ndarray:
    """Body-frame inertia (3x3) the drone's attitude actually feels, at the hanging pose.

    Takes the joint-space mass matrix at the model's default pose (pendulum hanging
    straight down) and eliminates every degree of freedom except the drone's three
    rotations (Schur complement). Translation and any free pendulum joint are allowed to
    move, while directions the pendulum cannot swing in stay rigidly coupled. For the
    default model this gives roll ~0.012 (drone only), pitch ~0.13 (drone + rigid
    payload), yaw ~0.022 kg m^2.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M, data.qM)
    dof = model.jnt_dofadr[model.joint(free_joint).id]
    rot = [dof + 3, dof + 4, dof + 5]  # free joint: 3 translational then 3 rotational dofs
    other = [i for i in range(model.nv) if i not in rot]
    J = M[np.ix_(rot, rot)] - M[np.ix_(rot, other)] @ np.linalg.solve(
        M[np.ix_(other, other)], M[np.ix_(other, rot)])
    return 0.5 * (J + J.T)


def bare_drone_params(model: mujoco.MjModel, body: str = "drone") -> tuple[float, np.ndarray]:
    """Mass (kg) and body-frame inertia (3x3) of the drone body alone, no payload.

    Use these as TrackingController3D(mass=..., inertia=...) to get a "regular drone"
    controller tuned as if nothing were attached.
    """
    bid = model.body(body).id
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(model.body_iquat[bid], dtype=float))
    R = R.reshape(3, 3)
    J = R @ np.diag(np.asarray(model.body_inertia[bid], dtype=float)) @ R.T
    return float(model.body_mass[bid]), J


class TrackingController3D(Controller):
    """Tracks a Trajectory with DronePendulumEnv (Dict observation).

    The controller keeps its own clock: the k-th call to act() (after reset()) uses the
    reference at t = k * env.control_dt.
    """

    def __init__(
        self,
        env,
        trajectory: Trajectory | None = None,
        kp=(4.0, 4.0, 6.0),
        kd=(3.0, 3.0, 4.0),
        k_att=(6.0, 6.0, 3.0),
        k_rate: float = 15.0,
        max_tilt: float = np.deg2rad(35.0),
        use_feedforward: bool = True,
        compensate_rate_loop: bool = True,
        rate_loop_gain: float | None = None,
        mass: float | None = None,
        inertia: np.ndarray | None = None,
        ki=(0.0, 0.0, 0.0),
        max_integral_acc: float = 3.0,
    ):
        """
        env : DronePendulumEnv, used for constants (mass, inertia, action scaling, dt).
        kp, kd : (x, y, z) position / velocity gains in acceleration units.
        k_att : (roll, pitch, yaw) attitude gains (1/s).
        k_rate : desired body-rate tracking bandwidth (1/s) for the rate-loop compensation.
        max_tilt : limit on the angle between the commanded thrust and vertical.
        use_feedforward : include acceleration + jerk feedforward (turn off to compare).
        compensate_rate_loop : apply step 6 (see module docstring). If False the desired
            rates are sent straight to the env.
        rate_loop_gain : proportional gain of the inner rate loop (N m per rad/s).
            Defaults to the env's actuation.rate_gain.
        mass, inertia : what the controller assumes about the vehicle. Default: total
            mass (drone + payload) and effective_rotational_inertia(model). Pass
            bare_drone_params(model) for a controller that doesn't know about the payload.
        ki : (x, y, z) integral gains on position error (1/s^3), off by default. Like an
            autopilot's integrator, it removes steady offsets such as a wrong mass.
        max_integral_acc : anti-windup clamp on the integral term (m/s^2).
        """
        env = env.unwrapped
        sim = env.simulator
        self.mass = float(mass) if mass is not None else sim.drone_model.total_mass
        self.g = float(-sim.model.opt.gravity[2])
        self.max_thrust = sim.actuation.max_thrust
        self.max_body_rate = sim.actuation.max_body_rate
        self.dt = env.control_dt
        self.inertia = (np.asarray(inertia, dtype=float) if inertia is not None
                        else effective_rotational_inertia(sim.model))
        self.rate_loop_gain = float(rate_loop_gain if rate_loop_gain is not None
                                    else sim.actuation.rate_gain)

        self.trajectory = trajectory or Hover()
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.k_att = np.asarray(k_att, dtype=float)
        self.k_rate = float(k_rate)
        self.max_tilt = float(max_tilt)
        self.use_feedforward = use_feedforward
        self.compensate_rate_loop = compensate_rate_loop
        self.ki = np.asarray(ki, dtype=float)
        self.max_integral_acc = float(max_integral_acc)

        self.t = 0.0
        self.integral_acc = np.zeros(3)
        self.last_ref: RefPoint | None = None
        self.last_debug: dict = {}

    def reset(self) -> None:
        self.t = 0.0
        self.integral_acc = np.zeros(3)
        self.last_ref = None
        self.last_debug = {}

    def thrust_to_action(self, thrust: float) -> float:
        return 2.0 * thrust / self.max_thrust - 1.0

    def act(self, obs, info=None):
        pos = np.asarray(obs["drone_position_delta"], dtype=float)  # relative to spawn
        vel = np.asarray(obs["drone_linear_velocity"], dtype=float)  # world frame
        omega = np.asarray(obs["drone_angular_velocity"], dtype=float)  # body frame
        R = quat_to_rotmat(obs["drone_orientation"])

        ref = self.trajectory(self.t)
        self.last_ref = ref
        self.t += self.dt
        a_ref = ref.acc if self.use_feedforward else np.zeros(3)
        j_ref = ref.jerk if self.use_feedforward else np.zeros(3)
        yaw_rate_ref = ref.yaw_rate if self.use_feedforward else 0.0

        # 1. desired force, with a tilt limit on its direction
        self.integral_acc = np.clip(self.integral_acc - self.ki * (pos - ref.pos) * self.dt,
                                    -self.max_integral_acc, self.max_integral_acc)
        a_cmd = (a_ref - self.kp * (pos - ref.pos) - self.kd * (vel - ref.vel)
                 + self.integral_acc)
        F = self.mass * (a_cmd + self.g * E3)
        F[2] = max(F[2], 0.1 * self.mass * self.g)
        horiz = np.linalg.norm(F[:2])
        max_horiz = F[2] * np.tan(self.max_tilt)
        tilt_clamped = horiz > max_horiz
        if tilt_clamped:
            F[:2] *= max_horiz / horiz
        F_norm = float(np.linalg.norm(F))

        # 2. thrust along the current body z
        thrust = float(np.clip(F @ R[:, 2], 0.0, self.max_thrust))

        # 3. desired attitude
        b3_d = F / F_norm
        b1_c = np.array([np.cos(ref.yaw), np.sin(ref.yaw), 0.0])
        b2_d = np.cross(b3_d, b1_c)
        b2_d /= np.linalg.norm(b2_d)
        b1_d = np.cross(b2_d, b3_d)
        R_d = np.column_stack([b1_d, b2_d, b3_d])

        # 4. attitude error in the body frame
        e_R = 0.5 * _vee(R_d.T @ R - R.T @ R_d)

        # 5. desired body rates: flatness feedforward + attitude feedback
        if tilt_clamped:
            w_d = np.array([0.0, 0.0, yaw_rate_ref * b3_d[2]])
        else:
            h = (self.mass / F_norm) * (j_ref - (b3_d @ j_ref) * b3_d)
            w_d = np.array([-h @ b2_d, h @ b1_d, yaw_rate_ref * b3_d[2]])
        omega_ff = R.T @ R_d @ w_d
        omega_des = omega_ff - self.k_att * e_R

        # 6. inertia-aware compensation of the inner P rate loop
        if self.compensate_rate_loop:
            J = self.inertia
            torque = J @ (self.k_rate * (omega_des - omega)) + np.cross(omega, J @ omega)
            omega_cmd = omega + torque / self.rate_loop_gain
        else:
            omega_cmd = omega_des

        self.last_debug = {"R_d": R_d, "thrust": thrust, "omega_ff": omega_ff,
                           "omega_des": omega_des, "omega_cmd": omega_cmd, "e_R": e_R}
        action = np.concatenate([[self.thrust_to_action(thrust)], omega_cmd / self.max_body_rate])
        return np.clip(action, -1.0, 1.0).astype(np.float32)
