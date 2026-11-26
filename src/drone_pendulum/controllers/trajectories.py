"""Reference trajectories for the tracking controllers.

Every trajectory is a function of time returning a `RefPoint` with position, velocity,
acceleration and jerk (all 3D, world frame) plus yaw and yaw rate. Positions are
RELATIVE TO THE DRONE'S SPAWN POINT, which is also what both envs report
(planar `obs[0:2]`, 3D `obs["drone_position_delta"]`), so a trajectory that starts at
the origin starts exactly where the drone is.

Jerk is needed because the tracking controllers turn it into body-rate feedforward
(differential flatness: thrust direction follows acceleration, body rates follow jerk).

The planar controller only uses the y and z components, so planar trajectories should
live in the y-z plane: pass `axes=(1, 2)` to the periodic shapes.

All periodic shapes start at the origin with zero velocity and ramp their speed up
smoothly over `ramp_time`, so there is no step in the commanded velocity at t = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RefPoint:
    pos: np.ndarray        # (3,) m
    vel: np.ndarray        # (3,) m/s
    acc: np.ndarray        # (3,) m/s^2
    jerk: np.ndarray       # (3,) m/s^3
    yaw: float = 0.0       # rad
    yaw_rate: float = 0.0  # rad/s


class Trajectory:
    """Base class. Subclasses implement sample(t)."""

    def sample(self, t: float) -> RefPoint:
        raise NotImplementedError

    def __call__(self, t: float) -> RefPoint:
        return self.sample(t)


def _zeros() -> np.ndarray:
    return np.zeros(3)


class Hover(Trajectory):
    """Hold a fixed point (default: the spawn point)."""

    def __init__(self, point=(0.0, 0.0, 0.0), yaw: float = 0.0):
        self.point = np.asarray(point, dtype=float)
        self.yaw = float(yaw)

    def sample(self, t):
        return RefPoint(self.point.copy(), _zeros(), _zeros(), _zeros(), self.yaw, 0.0)


# ----------------------------------------------------------------------------- helpers

def _quintic_blend(tau: float) -> tuple[float, float, float, float]:
    """Minimum-jerk blend sigma(tau) on [0, 1] and its first three tau-derivatives."""
    tau = float(np.clip(tau, 0.0, 1.0))
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    ddds = 60 - 360 * tau + 360 * tau**2
    return s, ds, dds, ddds


class _RampedPhase:
    """Phase s(t) whose rate ramps smoothly from 0 to omega over ramp_time.

    s_dot = omega * sigma(t / T) with sigma the quintic blend, so s, s_dot, s_ddot and
    s_dddot are all continuous. Returns (s, s_dot, s_ddot, s_dddot).
    """

    def __init__(self, omega: float, ramp_time: float):
        self.omega = float(omega)
        self.T = float(ramp_time)

    def __call__(self, t: float):
        w, T = self.omega, self.T
        if T <= 0.0:
            return w * t, w, 0.0, 0.0
        if t < T:
            tau = max(t, 0.0) / T
            # integral of the quintic blend from 0 to tau
            s = w * T * (2.5 * tau**4 - 3 * tau**5 + tau**6)
            sig, dsig, ddsig, _ = _quintic_blend(tau)
            return s, w * sig, w * dsig / T, w * ddsig / T**2
        return w * T * 0.5 + w * (t - T), w, 0.0, 0.0


class _PhasedPath(Trajectory):
    """A closed path p(s) traversed with a ramped phase s(t).

    Subclasses implement _path(s) -> (p, p', p'', p''') as (3,) arrays in world frame.
    Chain rule:
        v = p' s_dot
        a = p'' s_dot^2 + p' s_ddot
        j = p''' s_dot^3 + 3 p'' s_dot s_ddot + p' s_dddot
    """

    def __init__(self, period: float, ramp_time: float, center=(0.0, 0.0, 0.0),
                 yaw: float = 0.0):
        if period <= 0:
            raise ValueError("period must be positive")
        self.phase = _RampedPhase(2 * np.pi / period, ramp_time)
        self.offset = np.asarray(center, dtype=float)
        self.yaw = float(yaw)

    def _path(self, s):
        raise NotImplementedError

    def sample(self, t):
        s, sd, sdd, sddd = self.phase(t)
        p, dp, ddp, dddp = self._path(s)
        vel = dp * sd
        acc = ddp * sd**2 + dp * sdd
        jerk = dddp * sd**3 + 3 * ddp * sd * sdd + dp * sddd
        return RefPoint(self.offset + p, vel, acc, jerk, self.yaw, 0.0)


def _axis_vectors(axes) -> tuple[np.ndarray, np.ndarray]:
    a, b = axes
    e1, e2 = np.zeros(3), np.zeros(3)
    e1[a], e2[b] = 1.0, 1.0
    return e1, e2


class Circle(_PhasedPath):
    """Circle of `radius` through the origin, in the plane spanned by world `axes`.

    axes=(0, 1): horizontal circle in x-y (3D).  axes=(1, 2): vertical circle in y-z
    (planar). Starts at the origin heading along +axes[1].
    """

    def __init__(self, radius=0.5, period=4.0, axes=(0, 1), ramp_time=2.0, **kw):
        super().__init__(period, ramp_time, **kw)
        self.r = float(radius)
        self.e1, self.e2 = _axis_vectors(axes)

    def _path(self, s):
        r, e1, e2 = self.r, self.e1, self.e2
        c, sn = np.cos(s), np.sin(s)
        p = r * ((c - 1) * e1 + sn * e2)
        dp = r * (-sn * e1 + c * e2)
        ddp = r * (-c * e1 - sn * e2)
        dddp = r * (sn * e1 - c * e2)
        return p, dp, ddp, dddp


class Figure8(_PhasedPath):
    """Lemniscate of Gerono through the origin: A sin(s) along axes[0], (B/2) sin(2s)
    along axes[1]. One full figure-8 per `period` seconds."""

    def __init__(self, width=1.0, height=0.5, period=6.0, axes=(0, 1), ramp_time=2.0,
                 **kw):
        super().__init__(period, ramp_time, **kw)
        self.A, self.B = 0.5 * float(width), float(height)
        self.e1, self.e2 = _axis_vectors(axes)

    def _path(self, s):
        A, B, e1, e2 = self.A, self.B, self.e1, self.e2
        p = A * np.sin(s) * e1 + 0.5 * B * np.sin(2 * s) * e2
        dp = A * np.cos(s) * e1 + B * np.cos(2 * s) * e2
        ddp = -A * np.sin(s) * e1 - 2 * B * np.sin(2 * s) * e2
        dddp = -A * np.cos(s) * e1 - 4 * B * np.cos(2 * s) * e2
        return p, dp, ddp, dddp


class MinJerkWaypoints(Trajectory):
    """Rest-to-rest minimum-jerk segments through a list of waypoints.

    The drone starts at `start` (default origin), moves to each waypoint in turn taking
    `segment_time` seconds per leg (zero velocity/acceleration at every waypoint), then
    holds the last one. Good for point-to-point transport experiments.
    """

    def __init__(self, waypoints, segment_time=2.0, start=(0.0, 0.0, 0.0), yaw=0.0):
        pts = [np.asarray(start, dtype=float)] + [np.asarray(w, dtype=float) for w in waypoints]
        if any(p.shape != (3,) for p in pts):
            raise ValueError("waypoints must be 3D points")
        if segment_time <= 0:
            raise ValueError("segment_time must be positive")
        self.points = pts
        self.T = float(segment_time)
        self.yaw = float(yaw)

    def sample(self, t):
        n_seg = len(self.points) - 1
        if n_seg == 0 or t >= n_seg * self.T:
            return RefPoint(self.points[-1].copy(), _zeros(), _zeros(), _zeros(), self.yaw, 0.0)
        k = int(max(t, 0.0) // self.T)
        p0, p1 = self.points[k], self.points[k + 1]
        s, ds, dds, ddds = _quintic_blend((t - k * self.T) / self.T)
        d, T = p1 - p0, self.T
        return RefPoint(p0 + s * d, d * ds / T, d * dds / T**2, d * ddds / T**3,
                        self.yaw, 0.0)


def make_trajectory(name: str, planar: bool, **kwargs) -> Trajectory:
    """Convenience factory used by scripts: name in {hover, circle, figure8, waypoints}.

    For planar=True the periodic shapes are put in the y-z plane and waypoints are
    given as (y, z) pairs.
    """
    axes = (1, 2) if planar else (0, 1)
    if name == "hover":
        return Hover(**kwargs)
    if name == "circle":
        return Circle(axes=axes, **kwargs)
    if name == "figure8":
        return Figure8(axes=axes, **kwargs)
    if name == "waypoints":
        if planar:
            default = [(0.0, 0.5, 0.0), (0.0, 0.5, 0.5), (0.0, 0.0, 0.0)]
        else:
            default = [(0.5, 0.0, 0.0), (0.5, 0.5, 0.3), (0.0, 0.0, 0.0)]
        if planar and "waypoints" in kwargs:
            kwargs["waypoints"] = [(0.0, y, z) for y, z in kwargs["waypoints"]]
        kwargs.setdefault("waypoints", default)
        return MinJerkWaypoints(**kwargs)
    raise ValueError(f"unknown trajectory '{name}'")
