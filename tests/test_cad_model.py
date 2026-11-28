"""Tests for the hardware-CAD models and the controllers running on them."""

import gymnasium as gym
import numpy as np
import pytest

import drone_pendulum  # noqa: F401  (registers envs)
from drone_pendulum.controllers.lqr_2d import LQRController, linearize, solve_dare, _dare_residual
from drone_pendulum.controllers.trajectories import Hover, make_trajectory
from drone_pendulum.controllers.tracking_2d import PlanarTrackingController
from drone_pendulum.controllers.tracking_3d import (
    TrackingController3D, bare_drone_params, effective_rotational_inertia)
from drone_pendulum.envs.planar_drone_pendulum_env import PlanarDronePendulumEnv, PlanarEnvConfig
from drone_pendulum.envs.registration import CAD_MODEL_PATH, ENV_ID, PLANAR_CAD_MODEL_PATH


def planar_cad_env(seconds=8.0):
    return PlanarDronePendulumEnv(PlanarEnvConfig(
        model_path=str(PLANAR_CAD_MODEL_PATH), max_episode_seconds=seconds,
        init_joint_angle_range=0.0, init_joint_vel_range=0.0))


def test_cad_parameters_match_urdf():
    env = gym.make(ENV_ID, episode_limit=1, model_path=str(CAD_MODEL_PATH))
    model = env.unwrapped.simulator.model
    env.close()
    np.testing.assert_allclose(model.body("drone").mass, 0.6)
    np.testing.assert_allclose(model.body("link1").mass, 0.16460167956)
    np.testing.assert_allclose(model.body("link2").mass, 0.067)
    np.testing.assert_allclose(model.body_subtreemass[model.body("drone").id], 0.83160167956)

    p_env = planar_cad_env()
    m = p_env.sim.drone_model
    p_env.close()
    assert np.isclose(m.total_mass, 0.83160167956)
    assert np.isclose(m.link1_length, 0.100)
    assert np.isclose(m.link2_length, 0.075)


def test_cad_payload_is_rigid_about_roll():
    """CAD bearings rotate about body y: the payload is rigidly attached about body x."""
    env = gym.make(ENV_ID, episode_limit=1, model_path=str(CAD_MODEL_PATH))
    model = env.unwrapped.simulator.model
    env.close()
    J_eff = effective_rotational_inertia(model)
    _, J_bare = bare_drone_params(model)
    assert J_eff[0, 0] > 2.0 * J_bare[0, 0]          # roll drags the rigid payload
    assert J_eff[1, 1] < 1.1 * J_bare[1, 1]          # pitch: payload swings freely


def test_planar_cad_zero_action_hovers():
    env = planar_cad_env()
    env.reset(seed=0)
    for _ in range(100):  # 1 s
        obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    env.close()
    assert np.all(np.abs(obs[0:2]) < 0.01)


@pytest.mark.parametrize("name", ["circle", "figure8", "waypoints"])
def test_planar_tracking_on_cad(name):
    env = planar_cad_env()
    traj = make_trajectory(name, planar=True)
    ctrl = PlanarTrackingController(env, traj, ki=(1.0, 3.0))
    obs, info = env.reset(seed=0)
    ctrl.reset()
    errors = []
    while True:
        obs, _, term, trunc, info = env.step(ctrl.act(obs, info))
        errors.append(np.linalg.norm(obs[0:2] - traj(ctrl.t).pos[1:3]))
        if term or trunc:
            break
    env.close()
    settled = np.array(errors[len(errors) // 4:])
    assert np.sqrt(np.mean(settled**2)) < 0.01


@pytest.mark.parametrize("name", ["hover", "circle", "figure8"])
def test_3d_tracking_on_cad(name):
    env = gym.make(ENV_ID, episode_limit=400, model_path=str(CAD_MODEL_PATH))
    traj = Hover() if name == "hover" else make_trajectory(name, planar=False)
    ctrl = TrackingController3D(env, traj, ki=(1.0, 1.0, 3.0))
    obs, info = env.reset(seed=0)
    ctrl.reset()
    errors, terminated = [], False
    while True:
        obs, _, terminated, trunc, info = env.step(ctrl.act(obs, info))
        errors.append(np.linalg.norm(obs["drone_position_delta"] - traj(ctrl.t).pos))
        if terminated or trunc:
            break
    env.close()
    assert not terminated
    settled = np.array(errors[len(errors) // 2:])
    assert np.sqrt(np.mean(settled**2)) < 0.03


def test_dare_fallback_gives_a_valid_solution_on_cad():
    env = planar_cad_env()
    A, B = linearize(env, np.zeros(10), np.zeros(2))
    env.close()
    Q, R = np.eye(10), 10 * np.eye(2)
    P = solve_dare(A, B, Q, R)
    assert _dare_residual(A, B, Q, R, P) < 1e-6
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    assert np.max(np.abs(np.linalg.eigvals(A - B @ K))) < 1.0


def test_lqr_damps_push_on_cad():
    env = planar_cad_env(seconds=8.0)
    ctrl = LQRController.from_env(env)
    obs, info = env.reset(seed=0, options={"state": np.zeros(10)})
    pushed, states = False, []
    while True:
        if not pushed and info["time"] >= 1.0:
            env.sim.apply_disturbance("link2", [1.0, 0.0], 0.1)
            pushed = True
        obs, _, term, trunc, info = env.step(ctrl.act(obs, info))
        states.append(info["state"])
        if term or trunc:
            break
    env.close()
    x = np.array(states)
    phi = np.abs(np.c_[x[:, 2] + x[:, 3], x[:, 2] + x[:, 3] + x[:, 4]])
    assert np.rad2deg(phi.max()) > 20          # the push really swung the payload
    assert np.rad2deg(phi[-100:].max()) < 1.0  # and LQR damped it out
