"""Tests for the registered Gymnasium environment."""

import numpy as np
from gymnasium import make
from stable_baselines3.common.env_checker import check_env

from drone_pendulum.envs.registration import ENV_ID


def test_environment_truncates_at_episode_limit():
    env = make(ENV_ID, episode_limit=1)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(np.zeros(4, dtype=np.float32))
    assert not terminated
    assert truncated
    env.close()


def test_environment_terminates_when_safety_limit_is_exceeded():
    env = make(ENV_ID, episode_limit=10, max_position_error=0.0)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "position_limit"
    env.close()


def test_environment_passes_sb3_checker():
    env = make(ENV_ID, episode_limit=3)
    check_env(env.unwrapped, warn=True)
    env.close()
