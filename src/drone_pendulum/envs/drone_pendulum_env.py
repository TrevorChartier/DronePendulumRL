from typing import Optional
import numpy as np
import gymnasium as gym

from drone_pendulum.simulation.simulator import DronePendulumSimulator

class DronePendulumEnv(gym.Env):
    def __init__(self, model_path, max_episode_len):
        self.simulator = DronePendulumSimulator(model_path)
        
        self.initial_drone_pos = None
        self.curr_episode_step = None
        self.last_action = None
        self.max_episode_len = max_episode_len
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
        
        info = self._get_info()
        self.curr_episode_step = 0
        self.last_action = np.zeros(4, dtype=np.float32)
        
        return self._get_obs(), info
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (normalized Collective Thrust and Body Rate)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        self.simulator.step(action)
        self.curr_episode_step += 1
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        
        observation = self._get_obs()
        
        terminated = self._is_unsafe_state(observation)
        truncated = not terminated and self.curr_episode_step >= self.max_episode_len
        
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
        
    def _get_info(self):
        """TODO : Can return some info here that could be useful for debugging"""
        return {}
        
    def _is_unsafe_state(self, obs):
        """TODO: Right now just dummy method. We should implement some logic to end episodes if a crash is imminent"""
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