"""Quadratic propeller model for the Starling 2 Max (D0012).

Thrust and reaction torque follow the dynamometer fit in
agent_assets/starling2max_prop_coeffs.md. RPM limits come from
agent_assets/D0012_Starling_2_Max.params (VOXL_ESC_RPM_MIN/MAX).

The sim uses an instantaneous map: desired thrust → implied ω → RPM clip
→ actual F = kT·ω². Controllers still command thrust in N; dynamics backends
call `thrust_commands_to_rotor_thrusts` before writing MuJoCo ctrl.
"""

from __future__ import annotations

import numpy as np

# Dynamometer coefficients (N/(rad/s)^2 and N·m/(rad/s)^2).
THRUST_COEFF: float = 3.964738e-6
TORQUE_COEFF: float = 5.38697e-8

# ESC RPM limits from D0012.
RPM_MIN: float = 2000.0
RPM_MAX: float = 12000.0

# Prop angular velocity limits (rad/s).
OMEGA_MIN: float = RPM_MIN * (2.0 * np.pi / 60.0)
OMEGA_MAX: float = RPM_MAX * (2.0 * np.pi / 60.0)

# Constant τ/F ratio for the k·ω² model (used as MJCF motor yaw gear).
YAW_GEAR_RATIO: float = TORQUE_COEFF / THRUST_COEFF

# PX4 motor thrust model factor (THR_MDL_FAC in D0012 params).
PX4_THR_MDL_FAC: float = 0.9

# First-order rotor-speed time constants from D0012 / Gazebo motor model (seconds).
MOTOR_TIME_CONSTANT_UP_S: float = 0.055
MOTOR_TIME_CONSTANT_DOWN_S: float = 0.085


def max_thrust_per_rotor() -> float:
    """Return the maximum thrust (N) at OMEGA_MAX from the prop curve."""
    return float(THRUST_COEFF * OMEGA_MAX**2)


def hover_thrust_per_rotor(mass: float, g: float = 9.81) -> float:
    """Return per-rotor hover thrust (N) for a given take-off mass."""
    return float(mass * g / 4.0)


def rpm_to_omega(rpm: np.ndarray) -> np.ndarray:
    """Convert propeller RPM to angular velocity (rad/s)."""
    return np.asarray(rpm, dtype=float) * (2.0 * np.pi / 60.0)


def omega_to_rpm(omega: np.ndarray) -> np.ndarray:
    """Convert propeller angular velocity (rad/s) to RPM."""
    return np.asarray(omega, dtype=float) * (60.0 / (2.0 * np.pi))


def omega_to_thrust(omega: np.ndarray) -> np.ndarray:
    """Map prop angular velocity (rad/s) to thrust (N): F = kT·ω²."""
    omega = np.asarray(omega, dtype=float)
    return THRUST_COEFF * omega**2


def omega_to_torque(omega: np.ndarray) -> np.ndarray:
    """Map prop angular velocity (rad/s) to reaction torque (N·m): τ = kQ·ω²."""
    omega = np.asarray(omega, dtype=float)
    return TORQUE_COEFF * omega**2


def thrust_to_omega(thrust: np.ndarray) -> np.ndarray:
    """Map desired thrust (N) to prop angular velocity (rad/s) with RPM limits.

    Non-positive thrust maps to ω = 0 (motor off). Positive thrust inverts
    F = kT·ω² and clips ω to [OMEGA_MIN, OMEGA_MAX].
    """
    thrust = np.asarray(thrust, dtype=float)
    is_on = thrust > 0.0
    omega = np.zeros_like(thrust, dtype=float)
    omega[is_on] = np.sqrt(thrust[is_on] / THRUST_COEFF)
    omega[is_on] = np.clip(omega[is_on], OMEGA_MIN, OMEGA_MAX)
    return omega


def thrust_commands_to_rotor_thrusts(thrust_cmd: np.ndarray) -> np.ndarray:
    """Convert commanded per-rotor thrust (N) to actual thrust from the prop curve.

    Instantaneous model: ω = thrust_to_omega(F_cmd), then F = kT·ω². Supports
    shape (4,) or batched (N, 4). Actual thrust can differ from the command
    when RPM limits bind (upper saturation) or when a low command would imply
    ω below idle (floored at OMEGA_MIN).
    """
    omega = thrust_to_omega(thrust_cmd)
    return omega_to_thrust(omega)


def normalized_thrust_to_signal(
    normalized_thrust: np.ndarray,
    k: float = PX4_THR_MDL_FAC,
) -> np.ndarray:
    """Invert PX4 motor model T = (1-k)u + k·u² to recover signal u from T.

    Args:
        normalized_thrust: normalized thrust T; shape () or any broadcastable
            array. Values outside [0, 1] are clipped; negative/NaN map to 0.
        k: THR_MDL_FAC (default 0.9 for Starling 2 Max).

    Returns:
        Normalized actuator signal u in [0, 1], same shape as input.
    """
    thrust = np.asarray(normalized_thrust, dtype=float)
    thrust = np.nan_to_num(thrust, nan=0.0, posinf=1.0, neginf=0.0)
    thrust = np.clip(thrust, 0.0, 1.0)
    one_minus_k = 1.0 - k
    discriminant = one_minus_k**2 + 4.0 * k * thrust
    return (-one_minus_k + np.sqrt(discriminant)) / (2.0 * k)


def signal_to_rpm(signal: np.ndarray) -> np.ndarray:
    """Map normalized actuator signal u in [0, 1] to ESC RPM command.

    u=0 → RPM_MIN, u=1 → RPM_MAX (VOXL_ESC limits from D0012 params).
    """
    u = np.clip(np.asarray(signal, dtype=float), 0.0, 1.0)
    return RPM_MIN + u * (RPM_MAX - RPM_MIN)


def px4_normalized_thrust_to_omega(
    normalized_thrust: np.ndarray,
    k: float = PX4_THR_MDL_FAC,
) -> np.ndarray:
    """Convert PX4 per-motor normalized thrust T to prop angular velocity (rad/s).

    Pipeline: T → u (quadratic inverse) → RPM → ω. Supports shape (4,) or (N, 4).
    """
    signal = normalized_thrust_to_signal(normalized_thrust, k=k)
    rpm = signal_to_rpm(signal)
    return rpm_to_omega(rpm)


def first_order_motor_filter_step(
    previous_state: np.ndarray,
    commanded_state: np.ndarray,
    dt: float,
    time_constant_up: float = MOTOR_TIME_CONSTANT_UP_S,
    time_constant_down: float = MOTOR_TIME_CONSTANT_DOWN_S,
) -> np.ndarray:
    """Advance one asymmetric first-order motor filter step on rotor speed.

    Matches the Gazebo ``FirstOrderFilter::updateFilter`` used in PX4 SITL:
    spin-up and spin-down use separate time constants.

    Args:
        previous_state: filtered rotor speed (rad/s); shape (4,) or broadcastable.
        commanded_state: commanded rotor speed (rad/s); same shape.
        dt: elapsed time since the previous update (s).
        time_constant_up: spin-up time constant (s); 63.2% rise time.
        time_constant_down: spin-down time constant (s); 63.2% fall time.

    Returns:
        Updated filtered rotor speed, same shape as inputs.
    """
    if dt <= 0.0:
        return np.asarray(previous_state, dtype=float).copy()

    previous = np.asarray(previous_state, dtype=float)
    commanded = np.asarray(commanded_state, dtype=float)
    is_spinning_up = commanded > previous
    alpha_up = np.exp(-dt / time_constant_up)
    alpha_down = np.exp(-dt / time_constant_down)
    alpha = np.where(is_spinning_up, alpha_up, alpha_down)
    return alpha * previous + (1.0 - alpha) * commanded


def apply_first_order_motor_filter(
    commanded_omega: np.ndarray,
    times: np.ndarray,
    time_constant_up: float = MOTOR_TIME_CONSTANT_UP_S,
    time_constant_down: float = MOTOR_TIME_CONSTANT_DOWN_S,
    initial_omega: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the Gazebo first-order rotor-speed filter along a time series.

    Args:
        commanded_omega: target rotor speeds (rad/s); shape (N,) or (N, n_rotors).
        times: sample timestamps (s); shape (N,).
        time_constant_up: spin-up time constant (s).
        time_constant_down: spin-down time constant (s).
        initial_omega: optional initial filtered speed; defaults to zeros.

    Returns:
        Filtered rotor speeds, same shape as ``commanded_omega``.
    """
    commanded = np.asarray(commanded_omega, dtype=float)
    sample_times = np.asarray(times, dtype=float)
    if commanded.shape[0] != sample_times.shape[0]:
        raise ValueError("commanded_omega and times must have the same length")
    if commanded.ndim == 1:
        commanded = commanded.reshape(-1, 1)

    filtered = np.empty_like(commanded)
    if initial_omega is None:
        state = np.zeros(commanded.shape[1], dtype=float)
    else:
        initial = np.asarray(initial_omega, dtype=float)
        state = initial.reshape(-1) if initial.ndim == 1 else initial[0].copy()
        if state.size != commanded.shape[1]:
            raise ValueError("initial_omega must have one value per rotor")

    filtered[0] = state
    for index in range(1, commanded.shape[0]):
        dt = float(sample_times[index] - sample_times[index - 1])
        state = first_order_motor_filter_step(
            state,
            commanded[index],
            dt,
            time_constant_up=time_constant_up,
            time_constant_down=time_constant_down,
        )
        filtered[index] = state

    if np.asarray(commanded_omega, dtype=float).ndim == 1:
        return filtered[:, 0]
    return filtered


def px4_normalized_thrust_to_rotor_thrusts(
    normalized_thrust: np.ndarray,
    k: float = PX4_THR_MDL_FAC,
) -> np.ndarray:
    """Convert PX4 per-motor normalized thrust T to physical rotor thrust (N).

    Pipeline: T → u (quadratic inverse) → RPM → ω → F = kT·ω².
    Supports shape (4,) or batched (N, 4). Uses measured-motor path directly
    (not ``thrust_commands_to_rotor_thrusts``).

    Args:
        normalized_thrust: per-rotor T from ActuatorMotors.control[i].
        k: THR_MDL_FAC.

    Returns:
        Per-rotor thrust in Newtons, same shape as input.
    """
    return omega_to_thrust(px4_normalized_thrust_to_omega(normalized_thrust, k=k))


def px4_normalized_thrust_to_rotor_thrusts_with_dynamics(
    normalized_thrust: np.ndarray,
    times: np.ndarray,
    k: float = PX4_THR_MDL_FAC,
    time_constant_up: float = MOTOR_TIME_CONSTANT_UP_S,
    time_constant_down: float = MOTOR_TIME_CONSTANT_DOWN_S,
    initial_omega: np.ndarray | None = None,
) -> np.ndarray:
    """Convert PX4 motor commands to thrust after first-order rotor dynamics.

    Commands are mapped to target ω, filtered with asymmetric spin-up/down
    time constants, then converted to thrust via F = kT·ω².

    Args:
        normalized_thrust: per-rotor T; shape (N, 4).
        times: sample timestamps (s); shape (N,).
        k: THR_MDL_FAC.
        time_constant_up: spin-up time constant (s).
        time_constant_down: spin-down time constant (s).
        initial_omega: optional initial filtered ω per rotor.

    Returns:
        Per-rotor thrust in Newtons; shape (N, 4).
    """
    target_omega = px4_normalized_thrust_to_omega(normalized_thrust, k=k)
    filtered_omega = apply_first_order_motor_filter(
        target_omega,
        times,
        time_constant_up=time_constant_up,
        time_constant_down=time_constant_down,
        initial_omega=initial_omega,
    )
    return omega_to_thrust(filtered_omega)
