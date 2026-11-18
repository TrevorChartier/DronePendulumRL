# This adds the hover reward and failure limits to the existing planar environment.

from dataclasses import dataclass

import numpy as np

from drone_pendulum.envs.planar_drone_pendulum_env import PlanarDronePendulumEnv, PlanarEnvConfig


@dataclass
class HoverConfig(PlanarEnvConfig):
    init_pos_range: float = 0.05
    init_roll_range: float = np.deg2rad(5)
    position_scale: float = 0.25
    roll_scale: float = np.deg2rad(15)
    link_scale: float = np.deg2rad(20)
    velocity_scale: float = 1.0
    angular_velocity_scale: float = 2.0
    position_weight: float = 1.0
    roll_weight: float = 0.5
    link_weight: float = 0.5
    velocity_weight: float = 0.1
    angular_velocity_weight: float = 0.1
    action_weight: float = 0.005
    max_position_error: float = 1.0
    max_roll: float = np.deg2rad(45)
    max_link_angle: float = np.deg2rad(60)

    def __post_init__(self):
        for name in ("position_scale", "roll_scale", "link_scale", "velocity_scale",
                     "angular_velocity_scale", "max_position_error", "max_roll",
                     "max_link_angle", "max_episode_seconds", "control_hz"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("position_weight", "roll_weight", "link_weight", "velocity_weight",
                     "angular_velocity_weight", "action_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


class HoverEnv(PlanarDronePendulumEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config or HoverConfig(), render_mode=render_mode)
        self.metadata = {**self.metadata, "render_fps": round(1 / self.control_dt)}

    def failure_reason(self, obs):
        x = self.sim.drone_model.get_state_vector()
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(obs)):
            return "nonfinite_state"
        angles = np.arctan2(np.sin(np.cumsum(x[2:5])), np.cos(np.cumsum(x[2:5])))
        if np.any(np.abs(obs[:2]) > self.cfg.max_position_error):
            return "position_limit"
        if abs(angles[0]) > self.cfg.max_roll:
            return "roll_limit"
        if np.any(np.abs(angles[1:]) > self.cfg.max_link_angle):
            return "link_limit"
        return None

    def position_cost(self, obs):
        return float(np.sum((obs[:2] / self.cfg.position_scale) ** 2))

    def roll_cost(self):
        roll = self.sim.drone_model.get_drone_roll()
        return float((1 - np.cos(roll)) / (1 - np.cos(self.cfg.roll_scale)))

    def link_cost(self):
        angles = self.sim.drone_model.get_absolute_link_angles()
        return float(np.sum(1 - np.cos(angles)) / (1 - np.cos(self.cfg.link_scale)))

    def velocity_cost(self):
        velocity = self.sim.drone_model.get_drone_velocity()
        return float(np.sum((velocity / self.cfg.velocity_scale) ** 2))

    def angular_velocity_cost(self):
        model = self.sim.drone_model
        rates = np.r_[model.get_drone_roll_rate(), model.get_absolute_link_rates()]
        return float(np.sum((rates / self.cfg.angular_velocity_scale) ** 2))

    def action_cost(self, action):
        return float(np.mean(np.square(action)))

    def reward_terms(self, obs, action):
        c = self.cfg
        x = self.sim.drone_model.get_state_vector()
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(obs)):
            return {name: 0.0 for name in (
                "position_cost", "roll_cost", "link_cost", "velocity_cost",
                "angular_velocity_cost", "hover_bonus", "action_penalty")}
        terms = {
            "position_cost": c.position_weight * self.position_cost(obs),
            "roll_cost": c.roll_weight * self.roll_cost(),
            "link_cost": c.link_weight * self.link_cost(),
            "velocity_cost": c.velocity_weight * self.velocity_cost(),
            "angular_velocity_cost": c.angular_velocity_weight * self.angular_velocity_cost(),
        }
        terms["hover_bonus"] = float(np.exp(-sum(terms.values())))
        terms["action_penalty"] = c.action_weight * self.action_cost(action)
        return terms

    def _compute_reward(self, obs, action):
        terms = self.reward_terms(obs, action)
        return terms["hover_bonus"] - terms["action_penalty"]

    def _is_terminated(self, obs):
        return self.failure_reason(obs) is not None

    def _get_info(self):
        info = super()._get_info()
        info["failure_reason"] = self.failure_reason(self._get_obs())
        return info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("action must contain two finite values")
        obs, reward, terminated, truncated, info = super().step(action)
        info["reward_terms"] = self.reward_terms(obs, np.clip(action, -1, 1))
        # SB3 needs a finite terminal observation even if the physics failed.
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0), reward, terminated, truncated, info
