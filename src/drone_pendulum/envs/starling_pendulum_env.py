import gymnasium as gym
import numpy as np

from drone_pendulum.starling_simulation.controller import (
    LocalWaypointController,
    MotorDelayLocalWaypointController,
)
from drone_pendulum.starling_simulation.dynamics import StarlingPendulumDynamics
from drone_pendulum.starling_simulation.state import BatchedDroneState


class StarlingPendulumEnv(gym.Env):
    """Position-action environment using the identified Starling dynamics."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        model_path,
        policy_dt=0.1,
        max_episode_steps=300,
        render_mode=None,
    ):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.dynamics = StarlingPendulumDynamics(model_path)
        self.render_mode = render_mode
        self.policy_dt = float(policy_dt)
        self.control_steps = max(1, round(self.policy_dt / self.dynamics.sim_dt))
        self.max_episode_steps = max_episode_steps
        self.controller = MotorDelayLocalWaypointController(
            LocalWaypointController(
                self.dynamics.model,
                self.dynamics.drone_body_id,
                use_physical_mixer=True,
            ),
            self.dynamics.sim_dt,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, 0.1], dtype=np.float32),
            high=np.array([1.0, 1.0, 3.0], dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (24,), dtype=np.float32)
        self.state = None
        self.step_count = 0

    def _batched_state(self):
        return BatchedDroneState(
            drone_position=self.state.drone.position[None, :],
            drone_velocity=self.state.drone.velocity[None, :],
            drone_rotation=self.state.drone.rotation[None, :, :],
            drone_angular_velocity=self.state.drone.angular_velocity[None, :],
        )

    def _observation(self):
        q = self.state.joint_positions
        return np.asarray(
            np.concatenate([
                self.state.drone.position,
                self.state.drone.velocity,
                self.state.drone.rotation.reshape(-1),
                self.state.drone.angular_velocity,
                np.sin(q), np.cos(q), self.state.joint_velocities,
            ]),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.dynamics.reset()
        angles = self.np_random.uniform(-np.deg2rad(10), np.deg2rad(10), 2)
        velocities = self.np_random.uniform(-0.2, 0.2, 2)
        self.state = self.state.__class__(
            self.state.drone, angles, velocities, self.state.t
        )
        self.dynamics.set_state(self.state)
        self.controller.reset()
        self.step_count = 0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        for _ in range(self.control_steps):
            thrusts = self.controller.compute_control(
                self._batched_state(), action[None, :]
            )[0]
            self.state = self.dynamics.step(self.state, thrusts)
        self.step_count += 1
        observation = self._observation()
        pendulum_cost = np.sum((1.0 - observation[-4::2]) ** 2)
        velocity_cost = np.sum(self.state.joint_velocities ** 2)
        terminated = bool(np.linalg.norm(self.state.drone.position[2]) > 10)
        truncated = not terminated and self.step_count >= self.max_episode_steps
        return observation, float(-pendulum_cost - 0.1 * velocity_cost), terminated, truncated, {}

    def close(self):
        self.dynamics.close()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.dynamics.render()
