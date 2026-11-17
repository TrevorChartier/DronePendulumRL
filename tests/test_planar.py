"""Tests for the planar sim/env plumbing. These check wiring and basic physics sanity,
not your reward/controller design."""

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import drone_pendulum.envs  # noqa: F401
from drone_pendulum.envs.planar_drone_pendulum_env import OBS_DIM, PlanarDronePendulumEnv
from drone_pendulum.simulation.model_2d import STATE_DIM
from drone_pendulum.simulation.simulator_2d import PlanarDronePendulumSimulator


def test_model_dimensions():
    sim = PlanarDronePendulumSimulator()
    assert sim.drone_model.get_state_vector().shape == (STATE_DIM,)
    assert np.isclose(sim.drone_model.total_mass, 1.25)
    assert np.isclose(sim.drone_model.link1_length, 0.5)
    assert np.isclose(sim.drone_model.link2_length, 0.5)


def test_zero_action_hovers():
    """Zero action = hover thrust, so the drone should barely move from rest."""
    sim = PlanarDronePendulumSimulator()
    sim.reset()
    sim.step(np.zeros(2), n_substeps=1000)  # 2 s
    x = sim.drone_model.get_state_vector()
    assert np.all(np.abs(x) < 1e-3), x


def test_full_thrust_climbs_and_min_thrust_falls():
    sim = PlanarDronePendulumSimulator()
    sim.reset()
    sim.step(np.array([1.0, 0.0]), n_substeps=250)
    assert sim.drone_model.get_drone_velocity()[1] > 0.5
    sim.reset()
    sim.step(np.array([-1.0, 0.0]), n_substeps=250)
    assert sim.drone_model.get_drone_velocity()[1] < -0.5


def test_roll_rate_command_tracks():
    sim = PlanarDronePendulumSimulator()
    sim.reset()
    sim.step(np.array([0.0, 0.2]), n_substeps=200)
    target = 0.2 * sim.actuation.max_body_rate
    assert np.sign(sim.drone_model.get_drone_roll_rate()) == np.sign(target)


def test_set_and_get_state_roundtrip():
    sim = PlanarDronePendulumSimulator()
    x = np.arange(STATE_DIM, dtype=float) * 0.01
    sim.reset(x)
    assert np.allclose(sim.drone_model.get_state_vector(), x)


def test_disturbance_moves_payload():
    sim = PlanarDronePendulumSimulator()
    sim.reset()
    sim.apply_disturbance("link2", force_yz=[2.0, 0.0], duration=0.1)
    sim.step(np.zeros(2), n_substeps=100)
    phi = sim.drone_model.get_absolute_link_angles()
    assert np.abs(phi[1]) > 1e-2
    # disturbance must have expired and cleared
    assert np.allclose(sim.data.xfrc_applied[sim.drone_model.link2_id], 0.0)


def test_env_api():
    env = PlanarDronePendulumEnv()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs))


def test_env_truncates_at_max_length():
    env = gym.make("PlanarDronePendulum-v0")
    env.reset(seed=0)
    n = 0
    while True:
        n += 1
        *_, term, trunc, _ = env.step(np.zeros(2, dtype=np.float32))
        if term or trunc:
            break
    assert n == env.unwrapped.max_episode_steps
