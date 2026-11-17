"""Gymnasium environment for the planar drone + double pendulum.

    obs    : flat float32 vector (see OBS_NAMES)          shape (13,)
    action : [collective_thrust, roll_rate] in [-1, 1]    shape (2,)

The observation is a flat Box rather than a Dict so it plugs straight into standard RL
libraries (Stable-Baselines3, CleanRL, your own PPO/SAC code) without wrappers.

What is DONE here (plumbing):
    - simulator stepping at a fixed control rate (frame skip)
    - observation construction (angles encoded as sin/cos)
    - truncation at max episode length
    - rendering (human viewer / rgb_array)
    - a basic randomized reset

What is LEFT FOR YOU (search for "TODO(you)"):
    - _compute_reward            the reward function
    - _is_terminated             crash / failure detection
    - _sample_initial_state      initial-state distribution & domain randomization
    - _maybe_apply_disturbance   disturbance schedule for recovery experiments
"""

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from drone_pendulum.simulation.actuation_2d import ACTION_DIM
from drone_pendulum.simulation.model_2d import NQ, STATE_DIM
from drone_pendulum.simulation.simulator_2d import DEFAULT_MODEL_PATH, PlanarDronePendulumSimulator

OBS_NAMES = (
    "y_err", "z_err",
    "sin_roll", "cos_roll",
    "sin_q1", "cos_q1",
    "sin_q2", "cos_q2",
    "y_dot", "z_dot", "roll_dot", "q1_dot", "q2_dot",
)
OBS_DIM = len(OBS_NAMES)


@dataclass
class PlanarEnvConfig:
    """Knobs for the environment. Tweak these instead of editing the class."""

    model_path: str = DEFAULT_MODEL_PATH

    control_hz: float = 100.0      # policy rate; physics runs at 1/timestep (500 Hz)
    max_episode_seconds: float = 10.0

    # Initial-condition ranges used by the default _sample_initial_state
    init_pos_range: float = 0.0          # m, uniform +/- on y and z offset
    init_roll_range: float = 0.0         # rad
    init_joint_angle_range: float = np.deg2rad(10)   # rad, on q1 and q2
    init_joint_vel_range: float = 0.2    # rad/s, on q1_dot and q2_dot

    # Inner-loop / actuation parameters (forwarded to PlanarActuation)
    max_body_rate: float = 5.0
    rate_gain: float = 0.1


class PlanarDronePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config: PlanarEnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or PlanarEnvConfig()
        self.sim = PlanarDronePendulumSimulator(
            self.cfg.model_path,
            max_body_rate=self.cfg.max_body_rate,
            rate_gain=self.cfg.rate_gain,
        )

        self.frame_skip = max(1, int(round(1.0 / (self.cfg.control_hz * self.sim.dt))))
        self.control_dt = self.frame_skip * self.sim.dt
        self.max_episode_steps = int(round(self.cfg.max_episode_seconds / self.control_dt))

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)

        # Hover target in world (y, z). By default: wherever the drone spawned.
        self.target_yz = self.sim.drone_model.spawn_yz.copy()

        self.render_mode = render_mode
        self._viewer = None
        self._renderer = None

        self.step_count = 0
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # ================================================================== Gym API

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        # options["state"] lets evaluation scripts start from an exact 10-vector state.
        state = options.get("state")
        if state is None:
            state = self._sample_initial_state()
        self.sim.reset(np.asarray(state, dtype=float))

        self.step_count = 0
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self._maybe_apply_disturbance()
        self.sim.step(action, n_substeps=self.frame_skip)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = self._is_terminated(obs)
        truncated = (not terminated) and self.step_count >= self.max_episode_steps

        self.last_action = action
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), bool(terminated), bool(truncated), self._get_info()

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer
                self._viewer = mj_viewer.launch_passive(self.sim.model, self.sim.data)
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._viewer.cam.fixedcamid = self.sim.model.camera("side").id
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.sim.model, height=480, width=640)
            self._renderer.update_scene(self.sim.data, camera="side")
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ================================================================== helpers (done)

    def _get_obs(self) -> np.ndarray:
        x = self.sim.drone_model.get_state_vector()
        pos_err = self.sim.drone_model.get_drone_position() - self.target_yz
        roll, q1, q2 = x[2], x[3], x[4]
        obs = np.concatenate([
            pos_err,
            [np.sin(roll), np.cos(roll)],
            [np.sin(q1), np.cos(q1)],
            [np.sin(q2), np.cos(q2)],
            x[NQ:],
        ])
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        """Extra diagnostics (not seen by the policy). Useful for metrics/plots."""
        return {
            "state": self.sim.drone_model.get_state_vector(),
            "absolute_link_angles": self.sim.drone_model.get_absolute_link_angles(),
            "time": float(self.sim.data.time),
        }

    def hover_state(self) -> np.ndarray:
        """The equilibrium we want to stabilize: at target, level, links hanging down."""
        return np.zeros(STATE_DIM)

    # ================================================================== TODO(you)

    def _sample_initial_state(self) -> np.ndarray:
        """
        Initial-state distribution for training episodes.

        A simple uniform version is provided so the env runs. TODO(you):
          - decide what ranges make sense (start small, widen as training improves:
            this is a natural place for a curriculum)
          - domain randomization: payload mass, link lengths, actuation gain/delay.
            Mass/length changes mean editing self.sim.model (e.g. model.body_mass)
            before calling mj_forward. Keep a copy of the nominal values!
        """
        c = self.cfg
        rng = self.np_random
        x = np.zeros(STATE_DIM)
        x[0:2] = rng.uniform(-c.init_pos_range, c.init_pos_range, size=2)
        x[2] = rng.uniform(-c.init_roll_range, c.init_roll_range)
        x[3:5] = rng.uniform(-c.init_joint_angle_range, c.init_joint_angle_range, size=2)
        x[8:10] = rng.uniform(-c.init_joint_vel_range, c.init_joint_vel_range, size=2)
        return x

    def _maybe_apply_disturbance(self) -> None:
        """
        Called once per control step, before physics. TODO(you):
          Decide when/how to push the system, e.g. with some probability per episode,
          at a random time, push "link2" with a random force for ~0.05-0.2 s:

              self.sim.apply_disturbance("link2", force_yz=[F, 0.0], duration=0.1)

          For the proposal's disturbance-recovery evaluation you will want this to be
          controllable from outside (e.g. via reset options) so every controller sees
          the same disturbances.
        """
        pass

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        """
        TODO(you): design the reward. This is the most important design decision in the
        project. Things to think about:
          - which errors to penalize: drone position, roll, link angles (relative q's or
            absolute phi's from self.sim.drone_model.get_absolute_link_angles()?),
            velocities, control effort, action smoothness (action - self.last_action)
          - quadratic penalties vs. exp(-error) shaped bonuses
          - an alive bonus vs. a termination penalty: they interact with _is_terminated
            (a purely negative reward can teach the agent to crash early!)
          - scale: keep per-step rewards O(1) so value targets are well conditioned

        Returning 0.0 for now so the env runs end to end.
        """
        return 0.0

    def _is_terminated(self, obs: np.ndarray) -> bool:
        """
        TODO(you): end the episode on failure, e.g.
          - drone too far from target (|y_err|, |z_err| beyond a box)
          - drone flipped (|roll| too large)
          - NaNs / blow-up in the state
        Remember: termination != truncation. Termination means "no future reward"
        and is bootstrapped differently in PPO/SAC.
        """
        return False
