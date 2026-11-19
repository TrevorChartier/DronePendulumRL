# This checks that hover rewards the intended behavior and ends episodes at the agreed limits.

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from drone_pendulum.envs.hover_env import HoverConfig, HoverEnv


@pytest.fixture
def env():
    instance = HoverEnv()
    instance.reset(seed=0, options={"state": np.zeros(10)})
    yield instance
    instance.close()


def reward_at(env, state, action=None):
    env.sim.reset(state)
    return env._compute_reward(env._get_obs(), np.zeros(2) if action is None else action)


def test_hover_is_best_and_each_motion_costs(env):
    assert reward_at(env, np.zeros(10)) == pytest.approx(1)
    for index in range(10):
        state = np.zeros(10)
        state[index] = 0.1
        assert reward_at(env, state) < 1
    assert reward_at(env, np.zeros(10), np.ones(2)) < 1


def test_absolute_angles_and_rates(env):
    state = np.zeros(10)
    state[2:5] = [0.2, -0.2, 0]
    state[7:10] = [0.3, -0.3, 0]
    env.sim.reset(state)
    assert env.link_cost() == pytest.approx(0)
    state[2:5] = [0, 0.2, 0]
    env.sim.reset(state)
    assert env.link_cost() > 0
    assert env.angular_velocity_cost() > 0


@pytest.mark.parametrize("index,value,reason", [
    (0, 1.01, "position_limit"), (1, -1.01, "position_limit"),
    (2, np.deg2rad(46), "roll_limit"),
    (3, np.deg2rad(61), "link_limit"), (4, -np.deg2rad(61), "link_limit"),
])
def test_failures(env, index, value, reason):
    state = np.zeros(10)
    state[index] = value
    env.sim.reset(state)
    assert env.failure_reason(env._get_obs()) == reason
    assert env._is_terminated(env._get_obs())


def test_link_limit_uses_absolute_angle(env):
    state = np.zeros(10)
    state[3:5] = np.deg2rad([35, 35])
    env.sim.reset(state)
    assert env.failure_reason(env._get_obs()) == "link_limit"


def test_nonfinite_observation(env):
    obs = env._get_obs()
    obs[0] = np.nan
    assert env.failure_reason(obs) == "nonfinite_state"
    assert env._compute_reward(obs, np.zeros(2)) == 0


def test_time_limit_and_failure_are_distinct():
    env = HoverEnv(HoverConfig(max_episode_seconds=0.02))
    try:
        env.reset(options={"state": np.zeros(10)})
        assert env.step(np.zeros(2))[2:4] == (False, False)
        assert env.step(np.zeros(2))[2:4] == (False, True)
        state = np.zeros(10)
        state[0] = 1.1
        env.reset(options={"state": state})
        assert env.step(np.zeros(2))[2:4] == (True, False)
    finally:
        env.close()


def test_seeded_resets_and_gym_api(env):
    first, _ = env.reset(seed=8)
    second, _ = env.reset(seed=8)
    np.testing.assert_array_equal(first, second)
    check_env(env, skip_render_check=True)


def test_terms_reconstruct_reward(env):
    obs = env._get_obs()
    terms = env.reward_terms(obs, np.ones(2))
    assert env._compute_reward(obs, np.ones(2)) == pytest.approx(
        terms["hover_bonus"] - terms["action_penalty"])
