"""Run a controller in an env and record the trajectory (works for any Controller)."""

from dataclasses import dataclass, field

import numpy as np

from drone_pendulum.controllers.base import Controller


@dataclass
class Trajectory:
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))          # (T,)
    states: np.ndarray = field(default_factory=lambda: np.zeros((0, 10)))  # (T, 10)
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (T, 2)
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0))     # (T,)
    terminated: bool = False

    @property
    def episode_return(self) -> float:
        return float(np.sum(self.rewards))


def rollout(env, controller: Controller, seed: int | None = None, reset_options=None,
            disturbance=None) -> Trajectory:
    """
    Run one episode.

    disturbance : optional (time_s, body, force_yz, duration_s). Applied once when the
        sim clock passes time_s. Handy for repeatable push-recovery tests.
    """
    obs, info = env.reset(seed=seed, options=reset_options)
    controller.reset()

    ts, xs, us, rs = [info["time"]], [info["state"]], [], []
    pushed = disturbance is None
    terminated = truncated = False

    while not (terminated or truncated):
        if not pushed and info["time"] >= disturbance[0]:
            env.unwrapped.sim.apply_disturbance(*disturbance[1:])
            pushed = True
        u = controller.act(obs, info)
        obs, r, terminated, truncated, info = env.step(u)
        ts.append(info["time"])
        xs.append(info["state"])
        us.append(u)
        rs.append(r)

    return Trajectory(
        t=np.array(ts), states=np.array(xs), actions=np.array(us),
        rewards=np.array(rs), terminated=terminated,
    )
