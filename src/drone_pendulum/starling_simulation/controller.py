"""Quadrotor controllers used by the drone-soccer pipeline.

This module defines:

- `SE3Controller`: a stateless helper implementing the Lee et al. SE(3)
  geometric controller. It is *not* a `Controller` subclass because its
  natural goal is a 5-tuple ``(p_d, v_d, a_d, yaw_d, yaw_rate_d)`` that
  doesn't fit a single np.ndarray. Hand-coded demos call it directly.
- `LocalWaypointController`: vectorized mixed-frame waypoint controller.
  Takes a `BatchedWorldState` plus an ``(N, 3)`` action array
  ``(dx, dy, z_world)`` and returns per-env thrusts in one numpy pass.
  Used by both `DroneSoccerEnv` (N=1) and `WarpVecEnv` (N>1).

References:
    Lee, Leok, McClamroch, "Geometric Tracking Control of a Quadrotor UAV
    on SE(3)", CDC 2010.
    ModalAI Starling 2 Max MJCF: third_party/modalai_starling2max/starling2max.xml.
    Source params: agent_assets/D0012_Starling_2_Max.params (PX4 airframe).
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from drone_pendulum.starling_simulation.waypoint import compute_abs_waypoints
from drone_pendulum.starling_simulation.prop_model import (
    MOTOR_TIME_CONSTANT_DOWN_S,
    MOTOR_TIME_CONSTANT_UP_S,
    YAW_GEAR_RATIO,
    first_order_motor_filter_step,
    hover_thrust_per_rotor,
    max_thrust_per_rotor,
    omega_to_thrust,
    thrust_to_omega,
)
from drone_pendulum.starling_simulation.state import BatchedDroneState as BatchedWorldState
from drone_pendulum.starling_simulation.state import DroneState

# ---------------------------------------------------------------------------
# Quadrotor mixers used by SE3Controller.
#
# Each motor applies F_body = (0, 0, u_i) at its site, plus a pure yaw torque
# tau_z = gear_yaw_i * u_i. The body-frame wrench from r x F with
# r = (x_i, y_i, z_i) and F = (0, 0, u_i) is:
#   f_z   = sum u_i
#   tau_x = sum  y_i * u_i        ((r x F).x = y_i * u_i - z_i * 0)
#   tau_y = sum -x_i * u_i        ((r x F).y = z_i * 0   - x_i * u_i)
#   tau_z = sum gear_yaw_i * u_i  (the rotor yaw drag from the gear)
#
# Two geometry sets are kept:
#   - logfile: CA_ROTOR{0..3} from data/bv_px4_logs/2026-07-30/
#     01_21_03_position_mode.ulg (Starling 2 arm lengths, not Starling 2 Max).
#   - physical: agent_assets/starling2max.urdf + starling2max_prop_coeffs.md
#     (matches third_party/modalai_starling2max/starling2max.xml in sim).
#
# PX4 CA_ROTOR positions/KM are in body FRD (+y right). MuJoCo uses FLU (+y
# left), so y and KM signs flip when converting FRD → FLU.
# ---------------------------------------------------------------------------

# CA_ROTOR{0..3}_{PX,PY,KM} from 01_21_03_position_mode.ulg initial parameters.
_PX4_LOGFILE_ROTOR_PX = np.array([0.085, -0.085, 0.085, -0.085])
_PX4_LOGFILE_ROTOR_PY = np.array([0.0625, -0.0625, -0.0625, 0.0625])
_PX4_LOGFILE_ROTOR_KM = np.array([0.05, 0.05, -0.05, -0.05])

_LOGFILE_ROTOR_X = _PX4_LOGFILE_ROTOR_PX
_LOGFILE_ROTOR_Y = -_PX4_LOGFILE_ROTOR_PY
_LOGFILE_GEAR_YAW = -_PX4_LOGFILE_ROTOR_KM

# Starling 2 Max MJCF / URDF geometry (FLU body frame).
_PHYSICAL_ROTOR_X = np.array([0.095, -0.095, 0.095, -0.095])
_PHYSICAL_ROTOR_Y = np.array([-0.130, 0.130, 0.130, -0.130])
_PHYSICAL_GEAR_YAW = np.array([
    -YAW_GEAR_RATIO,
    -YAW_GEAR_RATIO,
    YAW_GEAR_RATIO,
    YAW_GEAR_RATIO,
])


def build_starling2max_mixer(*, use_physical_mixer: bool = False) -> np.ndarray:
    """Return the 4x4 matrix M such that [f_z, tau_x, tau_y, tau_z] = M @ u.

    Args:
        use_physical_mixer: when True, use Starling 2 Max MJCF geometry; when
            False (default), use CA_ROTOR values from the 2026-07-30 position
            mode ulog so the sim mixer matches logged PX4 flights.
    """
    if use_physical_mixer:
        rotor_x = _PHYSICAL_ROTOR_X
        rotor_y = _PHYSICAL_ROTOR_Y
        gear_yaw = _PHYSICAL_GEAR_YAW
    else:
        rotor_x = _LOGFILE_ROTOR_X
        rotor_y = _LOGFILE_ROTOR_Y
        gear_yaw = _LOGFILE_GEAR_YAW

    return np.vstack([
        np.ones(4),
        rotor_y,
        -rotor_x,
        gear_yaw,
    ])


def _vee(W: np.ndarray) -> np.ndarray:
    """so(3) vee: skew-symmetric 3x3 -> 3-vector."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def _body_inertia_tensor(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Return the 3x3 inertia tensor of a body expressed in *body* frame.

    MuJoCo stores body inertia as the diagonal in the body's principal-axes
    frame (model.body_inertia) plus the rotation from principal axes to body
    frame (model.body_iquat as a quaternion). When the principal axes are
    not aligned with the body axes (e.g. an asymmetric airframe), we must
    rotate the diagonal tensor before using it:
        I_body = R(iquat) diag(I_principal) R(iquat)^T.
    """
    iquat = np.asarray(model.body_iquat[body_id], dtype=float)
    R_flat = np.zeros(9)
    mujoco.mju_quat2Mat(R_flat, iquat)
    R = R_flat.reshape(3, 3)
    I_principal = np.diag(np.asarray(model.body_inertia[body_id], dtype=float))
    return R @ I_principal @ R.T


class SE3Controller:
    """Geometric SE(3) controller helper for the ModalAI Starling 2 Max.

    Stateless once constructed: every call to `compute_thrusts` does the
    Lee et al. CDC 2010 update from scratch. Drops the J*Omega_dot_des
    feedforward and uses naive per-rotor np.clip saturation (no priority
    allocation), but keeps the feedforward velocity, acceleration, and yaw
    rate. Mass and (rotated) inertia are read from the MuJoCo model so the
    controller stays consistent if the XML changes. Commands desired thrust;
    dynamics applies the quadratic prop model and RPM limits.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        drone_body_id: int,
        # Default position gains match PX4 MPC mode-4 accel feedback from
        # 01_21_03_position_mode.ulg (MPC_XY_P, MPC_Z_P, MPC_*_VEL_P_ACC).
        # kp/kv are acceleration-domain; ``compute_thrusts`` scales by mass.
        # Attitude/rate defaults from MC_* / MC_*RATE_* on the same log.
        kp: np.ndarray = np.array([0.95, 0.95, 5.0]),
        kv: np.ndarray = np.array([1.8, 1.8, 8.0]),
        k_R: np.ndarray = np.array([0.09, 0.09, 0.045]),
        k_Omega: np.ndarray = np.array([0.028, 0.028, 0.014]),
        T_max: float = max_thrust_per_rotor(),
        g: float = 9.81,
        use_physical_mixer: bool = False,
    ) -> None:
        """Build a controller bound to a drone body in a MuJoCo model.

        Args:
            model: MjModel containing the drone.
            drone_body_id: body id of the Starling 2 Max root body.
            kp, kv: world-frame position / velocity gains (3-vec, 1/s^2 and 1/s)
                in the same acceleration units as PX4 ``MPC_*_P`` and
                ``MPC_*_VEL_P_ACC``; multiplied by mass in ``compute_thrusts``.
            k_R, k_Omega: body-frame attitude / angular-rate gains (3-vec).
            T_max: per-rotor thrust upper bound from the prop curve at max RPM (N).
            g: gravity magnitude (m/s^2).
            use_physical_mixer: when True, use MJCF rotor geometry; default
                False uses the CA_ROTOR layout from the 2026-07-30 position log.
        """
        self._drone_id = int(drone_body_id)
        self._m = float(model.body_mass[self._drone_id])
        self._J = _body_inertia_tensor(model, self._drone_id)
        self._kp = np.asarray(kp, dtype=float)
        self._kv = np.asarray(kv, dtype=float)
        self._kR = np.asarray(k_R, dtype=float)
        self._kOmega = np.asarray(k_Omega, dtype=float)
        self._T_max = float(T_max)
        self._g = float(g)
        self._M_inv = np.linalg.inv(build_starling2max_mixer(use_physical_mixer=use_physical_mixer))

    @property
    def mass(self) -> float:
        return self._m

    @property
    def inertia(self) -> np.ndarray:
        return self._J

    def compute_thrusts(
        self,
        drone_state: DroneState,
        p_d: np.ndarray,
        v_d: np.ndarray,
        a_d: np.ndarray,
        yaw_d: float,
        yaw_rate_d: float,
    ) -> np.ndarray:
        """Compute per-rotor thrusts to track a full trajectory state.

        Args:
            drone_state: current drone state (position, velocity, rotation
                matrix, body-frame angular velocity).
            p_d, v_d, a_d: world-frame desired position (m), velocity (m/s),
                and acceleration (m/s^2).
            yaw_d: desired yaw in radians.
            yaw_rate_d: desired yaw rate in rad/s.

        Returns:
            np.ndarray of shape (4,), per-rotor thrust in N, clipped to
            [0, T_max].
        """
        p = drone_state.position
        v = drone_state.velocity
        R = drone_state.rotation
        Omega = drone_state.angular_velocity
        p_d = np.asarray(p_d, dtype=float)
        v_d = np.asarray(v_d, dtype=float)
        a_d = np.asarray(a_d, dtype=float)

        e_p = p - p_d
        e_v = v - v_d

        # Lee eq. 12 in acceleration form (matches PX4 MPC_POS_MODE=4), then
        # F = m a. Sign convention: F_des cancels gravity and tracks a_d.
        e3 = np.array([0.0, 0.0, 1.0])
        a_des = -self._kp * e_p - self._kv * e_v + self._g * e3 + a_d
        F_des = self._m * a_des

        # Total commanded thrust along the *current* body z. If body z drifts
        # away from F_des direction, f_total shrinks; the attitude loop will
        # rotate the body to realign.
        b3 = R[:, 2]
        f_total = float(F_des @ b3)

        # Build the desired body frame from F_des and yaw_d:
        #   b3_d  = F_des / |F_des|
        #   b1_d ~ projection of [cos yaw, sin yaw, 0] onto plane perp to b3_d
        F_norm = float(np.linalg.norm(F_des))
        if F_norm < 1e-6:
            # Degenerate: free-fall demand. Hold current attitude.
            R_d = R.copy()
        else:
            b3_d = F_des / F_norm
            b1_raw = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
            b2_d = np.cross(b3_d, b1_raw)
            n2 = float(np.linalg.norm(b2_d))
            if n2 < 1e-6:
                # b1_raw colinear with b3_d (drone pointing nearly straight up
                # along yaw axis): pick an arbitrary in-plane b1.
                b1_raw = np.array([-np.sin(yaw_d), np.cos(yaw_d), 0.0])
                b2_d = np.cross(b3_d, b1_raw)
                n2 = float(np.linalg.norm(b2_d))
            b2_d = b2_d / n2
            b1_d = np.cross(b2_d, b3_d)
            R_d = np.column_stack([b1_d, b2_d, b3_d])

        # Lee eq. 10/11: attitude and angular-rate errors.
        e_R = 0.5 * _vee(R_d.T @ R - R.T @ R_d)
        Omega_d_body = R.T @ np.array([0.0, 0.0, float(yaw_rate_d)])
        e_Omega = Omega - Omega_d_body

        # Lee eq. 16, dropping the J @ (...Omega_dot_des...) feedforward.
        tau_body = (
            -self._kR * e_R
            - self._kOmega * e_Omega
            + np.cross(Omega, self._J @ Omega)
        )

        wrench = np.array([f_total, tau_body[0], tau_body[1], tau_body[2]])
        u = self._M_inv @ wrench
        return np.clip(u, 0.0, self._T_max)


def _vee_batch(W: np.ndarray) -> np.ndarray:
    """Batched so(3) vee: ``(N, 3, 3)`` skew-symmetric -> ``(N, 3)``."""
    return np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], axis=-1)


class LocalWaypointController:
    """Vectorized mixed-frame waypoint controller for N parallel envs.

    Accepts a `BatchedWorldState` of N drones plus an ``(N, 3)`` array of
    mixed-frame ``(dx, dy, z_world)`` actions and returns ``(N, 4)``
    per-env, per-rotor thrusts in a single numpy pass. Same SE(3)
    geometric law as `SE3Controller`, with velocity / acceleration / yaw /
    yaw-rate feedforwards all zero. Yaw setpoint defaults to 0 for every
    env (`_DEFAULT_YAW`); to vary yaw per env, pass ``yaw_d`` as an
    ``(N,)`` array via `compute_control`.

    The xy displacement is in the *world* frame (no rotation by drone
    attitude). Intended for RL policies whose observations hide the drone's
    absolute xy but emit drone-relative xy displacements plus an absolute
    world altitude.
    """

    _DEFAULT_YAW: float = 0.0

    def __init__(
        self,
        model: mujoco.MjModel,
        drone_body_id: int,
        **se3_kwargs,
    ) -> None:
        """Build a local-waypoint controller bound to a model.

        Args:
            model: MjModel containing the drone (used to read mass / inertia).
            drone_body_id: body id of the Starling 2 Max root body.
            **se3_kwargs: forwarded to `SE3Controller` (gains, T_max, g, ...).
        """
        single = SE3Controller(model, drone_body_id, **se3_kwargs)
        self._mass = float(single.mass)
        self._inertia = np.asarray(single.inertia, dtype=float)
        self._kp = single._kp.astype(float)
        self._kv = single._kv.astype(float)
        self._kR = single._kR.astype(float)
        self._kOmega = single._kOmega.astype(float)
        self._T_max = float(single._T_max)
        self._g = float(single._g)
        self._M_inv = np.asarray(single._M_inv, dtype=float)

    def compute_control(
        self,
        states: BatchedWorldState,
        actions: np.ndarray,
        yaw_d: float | np.ndarray = _DEFAULT_YAW,
    ) -> np.ndarray:
        """Compute ``(N, 4)`` per-rotor thrusts for the batched state + actions.

        Args:
            states: batched world state (N envs).
            actions: shape ``(N, 3)`` mixed-frame action; xy is added to
                the drone's current world xy, z is absolute altitude.
            yaw_d: scalar (broadcast to every env) or shape ``(N,)`` per-env
                yaw setpoint (rad).

        Returns:
            ``(N, 4)`` per-rotor thrust in N, clipped to ``[0, T_max]``.
        """
        actions = np.asarray(actions, dtype=float)
        n = states.num_envs
        if actions.shape != (n, 3):
            raise ValueError(f"actions shape {actions.shape}, expected ({n}, 3)")

        yaw_d = np.broadcast_to(np.asarray(yaw_d, dtype=float), (n,)).copy()

        p = states.drone_position
        v = states.drone_velocity
        R = states.drone_rotation
        Omega = states.drone_angular_velocity

        # Mixed-frame goal: world xy is drone xy + action xy; z is absolute.
        p_d = compute_abs_waypoints(p, actions)

        # Position loop
        e_p = p - p_d
        v_des = -self._kp * e_p

        # Velocity loop
        e_v = v - v_des

        e3 = np.array([0.0, 0.0, 1.0])
        a_des = -self._kv * e_v + self._g * e3
        F_des = self._mass * a_des

        b3 = R[:, :, 2]
        f_total = np.einsum("ni,ni->n", F_des, b3)

        F_norm = np.linalg.norm(F_des, axis=-1)

        # Build the desired body frame, with safe fallbacks for the two
        # SE3Controller degenerate branches: free-fall demand (||F|| -> 0)
        # and b1_raw colinear with b3_d. We always compute both candidate
        # b1_raw vectors and select via np.where, so the math stays
        # branch-free over the batch.
        F_norm_safe = np.where(F_norm > 1e-6, F_norm, 1.0)
        b3_d = F_des / F_norm_safe[:, None]

        cos_yaw = np.cos(yaw_d)
        sin_yaw = np.sin(yaw_d)
        b1_raw_primary = np.stack([cos_yaw, sin_yaw, np.zeros_like(cos_yaw)], axis=-1)
        b1_raw_alt = np.stack([-sin_yaw, cos_yaw, np.zeros_like(cos_yaw)], axis=-1)

        b2_primary = np.cross(b3_d, b1_raw_primary)
        n2_primary = np.linalg.norm(b2_primary, axis=-1)
        use_alt = (n2_primary < 1e-6)[:, None]

        b2_alt = np.cross(b3_d, b1_raw_alt)
        b2_d = np.where(use_alt, b2_alt, b2_primary)
        n2 = np.linalg.norm(b2_d, axis=-1)
        n2_safe = np.where(n2 > 1e-6, n2, 1.0)
        b2_d = b2_d / n2_safe[:, None]
        b1_d = np.cross(b2_d, b3_d)
        R_d_normal = np.stack([b1_d, b2_d, b3_d], axis=-1)

        # Free-fall fallback: hold current attitude when F_des is degenerate.
        free_fall = (F_norm < 1e-6)[:, None, None]
        R_d = np.where(free_fall, R, R_d_normal)

        # SO(3) errors (Lee eq. 10/11), batched.
        R_T = np.transpose(R, (0, 2, 1))
        R_d_T = np.transpose(R_d, (0, 2, 1))
        skew = R_d_T @ R - R_T @ R_d
        e_R = 0.5 * _vee_batch(skew)

        # Omega_d_body = R^T @ [0, 0, yaw_rate_d] with yaw_rate_d=0, so
        # the body-frame desired angular velocity is identically zero and
        # e_Omega = Omega.
        e_Omega = Omega

        J_Omega = (self._inertia @ Omega[..., None]).squeeze(-1)
        cross_term = np.cross(Omega, J_Omega)
        tau_body = -self._kR * e_R - self._kOmega * e_Omega + cross_term

        wrench = np.concatenate([f_total[:, None], tau_body], axis=-1)
        u = wrench @ self._M_inv.T
        return np.clip(u, 0.0, self._T_max)

    @property
    def hover_thrust_per_rotor(self) -> float:
        """Return the steady per-rotor thrust that balances vehicle weight."""
        return hover_thrust_per_rotor(self._mass, self._g)


class MotorDelayLocalWaypointController:
    """Adds first-order rotor-speed dynamics to `LocalWaypointController`.

    `LocalWaypointController` (and `MujocoDynamics.step`) treat a commanded
    thrust as clipped and then realized on the very same sim step, but real
    ESC/motor pairs cannot change speed instantaneously; PX4 SITL models the
    lag with an asymmetric first-order filter on rotor speed (see
    `prop_model.apply_first_order_motor_filter`, which does the same thing
    offline from a logged rotor-speed series).
    This class applies that filter online, in thrust units, on the output of
    an inner `LocalWaypointController`: each `compute_control` call converts
    the commanded per-rotor thrust to a target rotor speed, advances the
    filter by one sim step, and converts the filtered speed back to thrust,
    so the returned thrust lags the commanded thrust by the motor's
    spin-up/spin-down time constant.

    Filter state is per-rotor angular velocity, shape ``(N, 4)`` for ``N``
    envs. New physical episodes initialize this state at hover speed because
    the simulated drone starts airborne. Goal-only transitions preserve the
    state. `WarpVecEnv` resets physical worlds individually, so `reset`
    accepts an optional per-env mask.
    """

    def __init__(
        self,
        controller: LocalWaypointController,
        sim_dt: float,
        time_constant_up: float = MOTOR_TIME_CONSTANT_UP_S,
        time_constant_down: float = MOTOR_TIME_CONSTANT_DOWN_S,
    ) -> None:
        """Wrap `controller` with a rotor-speed filter stepped at `sim_dt`.

        Args:
            controller: inner controller producing commanded per-rotor
                thrust (N), e.g. a `LocalWaypointController`.
            sim_dt: seconds between `compute_control` calls (the dynamics
                backend's sim timestep, not the coarser policy timestep).
            time_constant_up: spin-up time constant (s).
            time_constant_down: spin-down time constant (s).
        """
        self._controller = controller
        self._sim_dt = float(sim_dt)
        self._time_constant_up = float(time_constant_up)
        self._time_constant_down = float(time_constant_down)
        if self._sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")
        if self._time_constant_up <= 0.0 or self._time_constant_down <= 0.0:
            raise ValueError("motor time constants must be positive")

        self._hover_omega = float(
            thrust_to_omega(np.asarray(controller.hover_thrust_per_rotor))
        )
        self._filtered_omega: Optional[np.ndarray] = None

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        """Reset selected rotor-speed rows to the airborne hover state.

        Args:
            env_mask: boolean ``(N,)`` array selecting which envs to reset;
                None resets every env (the single-env case or a full batch
                reset).
        """
        if self._filtered_omega is None:
            return
        if env_mask is None:
            self._filtered_omega[:] = self._hover_omega
            return

        mask = np.asarray(env_mask, dtype=bool)
        if mask.shape != (self._filtered_omega.shape[0],):
            raise ValueError(
                f"env_mask shape {mask.shape}, expected "
                f"({self._filtered_omega.shape[0]},)"
            )
        self._filtered_omega[mask] = self._hover_omega

    def compute_control(
        self,
        states: BatchedWorldState,
        actions: np.ndarray,
        yaw_d: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Return ``(N, 4)`` per-rotor thrust after the motor-speed filter.

        Args:
            states, actions, yaw_d: forwarded to the inner controller's
                `compute_control` unchanged.

        Returns:
            ``(N, 4)`` per-rotor thrust in N: the inner controller's
            commanded thrust, delayed by one filtered rotor-speed step.
        """
        commanded_thrust = self._controller.compute_control(
            states, actions, yaw_d=yaw_d
        )
        commanded_omega = thrust_to_omega(commanded_thrust)

        if (
            self._filtered_omega is None
            or self._filtered_omega.shape != commanded_omega.shape
        ):
            self._filtered_omega = np.full_like(
                commanded_omega, self._hover_omega
            )

        self._filtered_omega = first_order_motor_filter_step(
            self._filtered_omega,
            commanded_omega,
            self._sim_dt,
            time_constant_up=self._time_constant_up,
            time_constant_down=self._time_constant_down,
        )
        return omega_to_thrust(self._filtered_omega)
