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
        episode_limit=1000,
        render_mode=None,
        max_position_error=2.0,
        min_altitude=0.2,
        max_altitude=3.0,
        max_tilt_angle=np.deg2rad(60.0),
        max_linear_velocity=10.0,
        max_angular_velocity=20.0,
        terminal_penalty=100.0,
    ):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.dynamics = StarlingPendulumDynamics(model_path)
        self.render_mode = render_mode
        self.policy_dt = float(policy_dt)
        self.control_steps = max(1, round(self.policy_dt / self.dynamics.sim_dt))
        self.episode_limit = episode_limit
        self.max_position_error = max_position_error
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.max_tilt_angle = max_tilt_angle
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.terminal_penalty = terminal_penalty
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
        self.reference_position = None
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
        self.reference_position = self.state.drone.position.copy()
        self.step_count = 0
        return self._observation(), {"termination_reason": None}

    def _safety_reason(self):
        position = self.state.drone.position
        if np.linalg.norm(position[:2] - self.reference_position[:2]) > self.max_position_error:
            return "position_limit"
        if not self.min_altitude <= position[2] <= self.max_altitude:
            return "altitude_limit"

        body_z_world = self.state.drone.rotation[:, 2]
        tilt_angle = np.arccos(np.clip(body_z_world[2], -1.0, 1.0))
        if tilt_angle > self.max_tilt_angle:
            return "tilt_limit"
        if np.linalg.norm(self.state.drone.velocity) > self.max_linear_velocity:
            return "linear_velocity_limit"
        if np.linalg.norm(self.state.drone.angular_velocity) > self.max_angular_velocity:
            return "angular_velocity_limit"
        return None

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        for _ in range(self.control_steps):
            thrusts = self.controller.compute_control(
                self._batched_state(), action[None, :]
            )[0]
            self.state = self.dynamics.step(self.state, thrusts)
        self.step_count += 1
        observation = self._observation()
        angle_cost = np.sum((1.0 - np.cos(self.state.joint_positions)) ** 2)
        pendulum_velocity_cost = np.sum(self.state.joint_velocities ** 2)
        position_error = self.state.drone.position - self.reference_position
        position_cost = np.sum(position_error ** 2)
        velocity_cost = np.sum(self.state.drone.velocity ** 2)
        tilt_angle = np.arccos(
            np.clip(self.state.drone.rotation[2, 2], -1.0, 1.0)
        )
        tilt_cost = tilt_angle ** 2
        action_cost = np.sum(action ** 2)

        reason = self._safety_reason()
        terminated = reason is not None
        truncated = not terminated and self.step_count >= self.episode_limit
        reward = -(
            5.0 * angle_cost
            + 0.1 * pendulum_velocity_cost
            + 0.5 * position_cost
            + 0.05 * velocity_cost
            + 0.5 * tilt_cost
            + 0.01 * action_cost
        )
        if terminated:
            reward -= self.terminal_penalty
        info = {
            "termination_reason": reason if terminated else (
                "timeout" if truncated else None
            ),
            "pendulum_angle_cost": float(angle_cost),
            "pendulum_velocity_cost": float(pendulum_velocity_cost),
            "position_cost": float(position_cost),
            "tilt_angle": float(tilt_angle),
        }
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.dynamics.close()

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return self.dynamics.render()
