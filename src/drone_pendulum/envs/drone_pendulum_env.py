from typing import Optional
import numpy as np
import gymnasium as gym

from drone_pendulum.simulation.simulator import DronePendulumSimulator

class DronePendulumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        model_path,
        episode_limit,
        render_mode: str | None = None,
        max_position_error: float = 2.0,
        max_tilt_angle: float = np.deg2rad(60.0),
        max_linear_velocity: float = 10.0,
        max_angular_velocity: float = 20.0,
    ):
        self.simulator = DronePendulumSimulator(model_path)
        self.n_substeps = 10  # physics steps per env step
        self.control_dt = self.n_substeps * float(self.simulator.model.opt.timestep)
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        
        self.initial_drone_pos = None
        self.curr_episode_step = None
        self.last_action = None
        self.episode_limit = episode_limit
        self.max_position_error = max_position_error
        self.max_tilt_angle = max_tilt_angle
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.termination_reason = None
        self.observation_space = gym.spaces.Dict(
            {
                "drone_position_delta": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32), # position relative to initial/desired point
                "drone_orientation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32), # quaternion
                "drone_linear_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "drone_angular_velocity": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "pendulum_joint_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32), # [sin(theta_1), cos(theta_1), sin(theta_2), cos(theta_2)]
                "pendulum_joint_angular_velocities": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
            }
        )
        
        # Action space: [collective_thrust, roll_rate, pitch_rate, yaw_rate] (all normalized)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused)

        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)

        angle_range = np.deg2rad(10)
        joint_positions = self.np_random.uniform(
            -angle_range,
            angle_range,
            size=2
        )
        velocity_range = 0.2
        joint_velocities = self.np_random.uniform(
            -velocity_range,
            velocity_range,
            size=2
        )
        
        self.simulator.reset(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
        )
        
        self.initial_drone_pos = (
            self.simulator.get_state()["drone_position"].copy()
        )
        
        self.curr_episode_step = 0
        self.last_action = np.zeros(4, dtype=np.float32)
        self.termination_reason = None
        info = self._get_info()
        
        return self._get_obs(), info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (normalized Collective Thrust and Body Rate)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        self.simulator.step(action, n_substeps=self.n_substeps)
        self.curr_episode_step += 1
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        
        observation = self._get_obs()
        
        terminated = self._is_unsafe_state(observation)
        truncated = not terminated and self.curr_episode_step >= self.episode_limit
        
        reward = self._calc_reward(observation)
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        
        state = self.simulator.get_state()
        sin_joint_pos = np.sin(state["joint_positions"])
        cos_joint_pos = np.cos(state["joint_positions"])
        return {
            "drone_position_delta": np.asarray(
                state["drone_position"] - self.initial_drone_pos, dtype=np.float32
            ),
            "drone_orientation": np.asarray(
                state["drone_orientation"], dtype=np.float32
            ),
            "drone_linear_velocity": np.asarray(
                state["drone_linear_velocity"], dtype=np.float32
            ),
            "drone_angular_velocity": np.asarray(
                state["drone_angular_velocity"], dtype=np.float32
            ),
            "pendulum_joint_pos": np.asarray(
                [sin_joint_pos[0], cos_joint_pos[0],
                sin_joint_pos[1], cos_joint_pos[1]],
                dtype=np.float32
            ),
            "pendulum_joint_angular_velocities": np.asarray(
                state["joint_velocities"], dtype=np.float32
            )
        }

    def render(self):
        """Render the current state as an RGB frame when enabled."""
        if self.render_mode != "rgb_array":
            return None
        return self.simulator.render()

    def close(self):
        """Release simulator resources."""
        self.simulator.close()
        
    def _get_info(self):
        """Return episode diagnostics useful during training and evaluation."""
        return {
            "termination_reason": self.termination_reason,
        }
        
    def _is_unsafe_state(self, obs):
        """Terminate episodes when the state leaves a safe operating envelope."""
        position_error = float(np.linalg.norm(obs["drone_position_delta"]))
        if position_error > self.max_position_error:
            self.termination_reason = "position_limit"
            return True

        quaternion = obs["drone_orientation"]
        scalar_part = float(np.clip(abs(quaternion[0]), 0.0, 1.0))
        tilt_angle = 2.0 * np.arccos(scalar_part)
        if tilt_angle > self.max_tilt_angle:
            self.termination_reason = "tilt_limit"
            return True

        if np.linalg.norm(obs["drone_linear_velocity"]) > self.max_linear_velocity:
            self.termination_reason = "linear_velocity_limit"
            return True

        if np.linalg.norm(obs["drone_angular_velocity"]) > self.max_angular_velocity:
            self.termination_reason = "angular_velocity_limit"
            return True

        self.termination_reason = None
        return False
    
    def _calc_reward(self, obs):
        """TODO: Probably will need to tune this and give it much more thought"""
        w_position = 1.0
        w_orientation = 1.0
        w_linear_velocity = 0.1
        w_angular_velocity = 0.05
        w_pendulum_angle = 5.0
        w_pendulum_velocity = 0.1
        w_action = 0.01

        position_error = np.sum(obs["drone_position_delta"] ** 2)

        # Assuming [w, x, y, z] and level orientation is identity
        q = obs["drone_orientation"]
        orientation_error = 1.0 - q[0] ** 2

        linear_velocity_error = np.sum(obs["drone_linear_velocity"] ** 2)
        angular_velocity_error = np.sum(obs["drone_angular_velocity"] ** 2)

        joint_pos = obs["pendulum_joint_pos"]
        pendulum_angle_error = (1.0 - joint_pos[1]) + (1.0 - joint_pos[3])

        pendulum_velocity_error = np.sum(
            obs["pendulum_joint_angular_velocities"] ** 2
        )

        # Control effort (normalized action)
        action_error = np.sum(self.last_action ** 2)

        reward = -(
            w_position * position_error
            + w_orientation * orientation_error
            + w_linear_velocity * linear_velocity_error
            + w_angular_velocity * angular_velocity_error
            + w_pendulum_angle * pendulum_angle_error
            + w_pendulum_velocity * pendulum_velocity_error
            + w_action * action_error
        )

        return float(reward)
