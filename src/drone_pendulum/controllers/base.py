"""Common interface for every controller (classical baselines and RL policies).

Controllers receive the env observation AND the info dict, so classical controllers can
use the true state (info["state"], the 10-vector) while RL policies use only `obs`.
This keeps evaluation code identical for LQR, PD and RL:

    obs, info = env.reset()
    controller.reset()
    while True:
        action = controller.act(obs, info)
        obs, r, term, trunc, info = env.step(action)
"""

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    def reset(self) -> None:
        """Clear any internal state (integrators, history buffers, RNN state...)."""

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Return a normalized action in [-1, 1]^action_dim."""


class ZeroController(Controller):
    """Always outputs zero, which is open-loop hover thrust and zero roll rate.

    Useful as a sanity check: with no disturbance and links hanging straight down the
    drone should stay put; otherwise you get to watch the passive dynamics.
    """

    def __init__(self, action_dim: int = 2):
        self.action_dim = action_dim

    def act(self, obs, info):
        return np.zeros(self.action_dim, dtype=np.float32)


class RandomController(Controller):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs, info):
        return self.action_space.sample()
