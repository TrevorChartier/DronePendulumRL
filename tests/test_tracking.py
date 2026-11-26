"""Tests for reference trajectories and the planar / 3D tracking controllers."""

import gymnasium as gym
import numpy as np
import pytest

import drone_pendulum  # noqa: F401  (registers envs)
from drone_pendulum.controllers.trajectories import (
    Circle, Figure8, Hover, MinJerkWaypoints, make_trajectory,
)
from drone_pendulum.controllers.tracking_2d import PlanarTrackingController
from drone_pendulum.controllers.tracking_3d import TrackingController3D
from drone_pendulum.envs.planar_drone_pendulum_env import PlanarDronePendulumEnv, PlanarEnvConfig
from drone_pendulum.controllers.tracking_3d import effective_rotational_inertia
from drone_pendulum.envs.registration import ENV_ID

TRAJECTORIES = [
    Circle(radius=0.5, period=4.0, axes=(0, 1), ramp_time=2.0),
    Circle(radius=0.3, period=3.0, axes=(1, 2), ramp_time=1.0),
    Figure8(width=1.0, height=0.5, period=6.0, ramp_time=2.0),
    MinJerkWaypoints([(0.5, 0.0, 0.2), (0.0, 0.5, 0.0)], segment_time=2.0),
]


@pytest.mark.parametrize("traj", TRAJECTORIES)
def test_trajectory_derivatives_are_consistent(traj):
    """vel/acc/jerk must match finite differences of pos/vel/acc (incl. the ramp)."""
    h = 1e-5
    for t in np.linspace(0.05, 7.0, 40):
        a, b = traj(t - h), traj(t + h)
        mid = traj(t)
        np.testing.assert_allclose((b.pos - a.pos) / (2 * h), mid.vel, atol=1e-5)
        np.testing.assert_allclose((b.vel - a.vel) / (2 * h), mid.acc, atol=1e-4)
        np.testing.assert_allclose((b.acc - a.acc) / (2 * h), mid.jerk, atol=1e-3)


@pytest.mark.parametrize("traj", TRAJECTORIES)
def test_trajectories_start_at_rest_at_origin(traj):
    p0 = traj(0.0)
    np.testing.assert_allclose(p0.pos, 0.0, atol=1e-12)
    np.testing.assert_allclose(p0.vel, 0.0, atol=1e-12)


def _run_planar(traj, seconds=8.0, **ctrl_kw):
    env = PlanarDronePendulumEnv(PlanarEnvConfig(
        max_episode_seconds=seconds, init_joint_angle_range=0.0, init_joint_vel_range=0.0))
    ctrl = PlanarTrackingController(env, traj, **ctrl_kw)
    obs, info = env.reset(seed=0)
    ctrl.reset()
    errors, actions = [], []
    while True:
        u = ctrl.act(obs, info)
        obs, _, term, trunc, info = env.step(u)
        errors.append(np.linalg.norm(obs[0:2] - traj(ctrl.t).pos[1:3]))
        actions.append(u)
        if term or trunc:
            break
    env.close()
    return np.array(errors), np.array(actions)


def test_planar_hover_holds_position():
    errors, _ = _run_planar(Hover(), seconds=3.0)
    assert errors.max() < 1e-3


def test_planar_tracks_vertical_circle():
    errors, actions = _run_planar(make_trajectory("circle", planar=True, radius=0.5, period=4.0))
    settled = errors[len(errors) // 4:]
    assert np.sqrt(np.mean(settled**2)) < 0.05
    assert np.all(np.abs(actions) <= 1.0)


def test_planar_feedforward_improves_tracking():
    traj = make_trajectory("figure8", planar=True)
    with_ff, _ = _run_planar(traj)
    without_ff, _ = _run_planar(traj, use_feedforward=False)
    assert np.mean(with_ff**2) < 0.25 * np.mean(without_ff**2)


def _run_3d(traj, steps=400, **ctrl_kw):
    env = gym.make(ENV_ID, episode_limit=steps)
    ctrl = TrackingController3D(env, traj, **ctrl_kw)
    obs, info = env.reset(seed=0)
    ctrl.reset()
    errors = []
    terminated = False
    while True:
        obs, _, terminated, trunc, info = env.step(ctrl.act(obs, info))
        errors.append(np.linalg.norm(obs["drone_position_delta"] - traj(ctrl.t).pos))
        if terminated or trunc:
            break
    q = obs["drone_orientation"]
    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
    env.close()
    return np.array(errors), terminated, yaw


def test_effective_inertia_reflects_single_axis_hinges():
    """Hinges rotate about body x: the payload is free in roll but rigid in pitch."""
    env = gym.make(ENV_ID, episode_limit=1)
    J = effective_rotational_inertia(env.unwrapped.simulator.model)
    env.close()
    np.testing.assert_allclose(J, J.T)
    assert J[1, 1] > 5 * J[0, 0]   # pitch drags the whole rigid payload
    assert J[0, 0] < 0.02          # roll: about the bare drone's inertia


def test_3d_hover_holds_position():
    errors, terminated, _ = _run_3d(Hover())
    assert not terminated
    assert errors.max() < 0.02


def test_3d_without_rate_compensation_fails_to_pitch():
    """Documents why the compensation exists: raw rate commands can't move along x."""
    _, terminated, _ = _run_3d(make_trajectory("waypoints", planar=False),
                               compensate_rate_loop=False)
    assert terminated


def test_3d_tracks_yaw_setpoint():
    errors, terminated, yaw = _run_3d(Hover(yaw=0.5), steps=300)
    assert not terminated
    assert abs(yaw - 0.5) < 0.02
    assert errors.max() < 0.02


@pytest.mark.parametrize("name", ["circle", "figure8", "waypoints"])
def test_3d_tracks_trajectories(name):
    errors, terminated, _ = _run_3d(make_trajectory(name, planar=False))
    assert not terminated
    settled = errors[len(errors) // 4:]
    assert np.sqrt(np.mean(settled**2)) < 0.02


def test_bare_drone_params_exclude_payload():
    from drone_pendulum.controllers.tracking_3d import bare_drone_params
    env = gym.make(ENV_ID, episode_limit=1)
    sim = env.unwrapped.simulator
    mass, J = bare_drone_params(sim.model)
    env.close()
    assert mass < sim.drone_model.total_mass
    assert J[1, 1] < 0.02  # no rigid payload in pitch


def test_integral_action_removes_mass_mismatch_offset():
    from drone_pendulum.controllers.tracking_2d import bare_drone_mass
    env = PlanarDronePendulumEnv(PlanarEnvConfig(
        max_episode_seconds=10.0, init_joint_angle_range=0.0, init_joint_vel_range=0.0))
    for ki, expect_offset in (((0.0, 0.0), True), ((1.0, 3.0), False)):
        ctrl = PlanarTrackingController(env, Hover(), mass=bare_drone_mass(env), ki=ki)
        obs, info = env.reset(seed=0)
        ctrl.reset()
        for _ in range(1000):
            obs, *_ = env.step(ctrl.act(obs, info))
        assert (abs(obs[1]) > 0.2) == expect_offset  # z sag from the underestimated mass
    env.close()
