# This loads a trained hover policy and applies the observation scaling it trained with.

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_pendulum.controllers.base import Controller
from drone_pendulum.envs.hover_env import HoverEnv


class PPOHoverController(Controller):
    def __init__(self, model_path, normalization_path, config):
        base = DummyVecEnv([lambda: HoverEnv(config)])
        try:
            self.normalizer = VecNormalize.load(str(normalization_path), base)
        except Exception:
            base.close()
            raise
        self.normalizer.training = False
        self.normalizer.norm_reward = False
        try:
            self.model = PPO.load(str(model_path))
        except Exception:
            self.normalizer.close()
            raise

    def act(self, obs, info):
        normalized = self.normalizer.normalize_obs(np.asarray(obs).copy())
        action, _ = self.model.predict(normalized, deterministic=True)
        return action

    def close(self):
        self.normalizer.close()
