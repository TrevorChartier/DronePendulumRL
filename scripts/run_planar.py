"""Run a controller on the planar drone-pendulum env, optionally view it and plot it.

Examples (from the repo root):
    python scripts/run_planar.py                          # zero action, plot
    python scripts/run_planar.py --render                 # watch it in the MuJoCo viewer
    python scripts/run_planar.py --push 3.0               # 3 N shove on the bottom mass at t=1s
    python scripts/run_planar.py --controller lqr --push 3.0
    python scripts/run_planar.py --q1 30 --q2 -20         # start with links at 30/-20 deg
"""

import argparse
import time

import numpy as np

from drone_pendulum.controllers.base import RandomController, ZeroController
from drone_pendulum.envs.planar_drone_pendulum_env import PlanarDronePendulumEnv, PlanarEnvConfig
from drone_pendulum.evaluation.rollout import rollout
from drone_pendulum.simulation.model_2d import STATE_NAMES


def make_controller(name, env):
    if name == "zero":
        return ZeroController()
    if name == "random":
        return RandomController(env.action_space)
    if name == "lqr":
        from drone_pendulum.controllers.lqr_2d import LQRController
        return LQRController.from_env(env)
    if name == "pd":
        from drone_pendulum.controllers.cascaded_pd_2d import CascadedPDController
        return CascadedPDController(env)
    raise ValueError(name)


class _RealTime:
    """Wraps a controller to slow the loop to wall-clock speed when rendering."""

    def __init__(self, inner, dt):
        self.inner, self.dt, self._t = inner, dt, None

    def reset(self):
        self.inner.reset()
        self._t = time.perf_counter()

    def act(self, obs, info):
        self._t += self.dt
        time.sleep(max(0.0, self._t - time.perf_counter()))
        return self.inner.act(obs, info)


def plot(traj):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    t, x = traj.t, traj.states
    axes[0].plot(t, x[:, 0], label="y"); axes[0].plot(t, x[:, 1], label="z")
    axes[0].set_ylabel("drone pos offset [m]")
    axes[1].plot(t, np.rad2deg(x[:, 2]), label="roll")
    axes[1].set_ylabel("drone roll [deg]")
    phi1 = x[:, 2] + x[:, 3]
    phi2 = phi1 + x[:, 4]
    axes[2].plot(t, np.rad2deg(phi1), label="phi1 (abs)")
    axes[2].plot(t, np.rad2deg(phi2), label="phi2 (abs)")
    axes[2].set_ylabel("link angle [deg]")
    if len(traj.actions):
        axes[3].step(t[1:], traj.actions[:, 0], label="thrust (norm)")
        axes[3].step(t[1:], traj.actions[:, 1], label="roll rate (norm)")
    axes[3].set_ylabel("action"); axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
    fig.suptitle(f"return = {traj.episode_return:.2f}   terminated = {traj.terminated}")
    fig.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default="zero", choices=["zero", "random", "lqr", "pd"])
    p.add_argument("--render", action="store_true", help="open the MuJoCo viewer")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--q1", type=float, default=None, help="initial q1 in degrees")
    p.add_argument("--q2", type=float, default=None, help="initial q2 in degrees")
    p.add_argument("--push", type=float, default=0.0, help="sideways force [N] on link2")
    p.add_argument("--push-time", type=float, default=1.0)
    p.add_argument("--push-duration", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = PlanarDronePendulumEnv(
        PlanarEnvConfig(max_episode_seconds=args.seconds),
        render_mode="human" if args.render else None,
    )
    controller = make_controller(args.controller, env)
    if args.render:
        controller = _RealTime(controller, env.control_dt)

    reset_options = None
    if args.q1 is not None or args.q2 is not None:
        x0 = env.hover_state()
        x0[3] = np.deg2rad(args.q1 or 0.0)
        x0[4] = np.deg2rad(args.q2 or 0.0)
        reset_options = {"state": x0}

    disturbance = None
    if args.push:
        disturbance = (args.push_time, "link2", [args.push, 0.0], args.push_duration)

    traj = rollout(env, controller, seed=args.seed, reset_options=reset_options,
                   disturbance=disturbance)
    env.close()

    print(f"steps={len(traj.rewards)}  return={traj.episode_return:.3f}  "
          f"terminated={traj.terminated}")
    print("final state:")
    for name, v in zip(STATE_NAMES, traj.states[-1]):
        print(f"  {name:>9s} = {v: .4f}")

    if not args.no_plot:
        plot(traj)


if __name__ == "__main__":
    main()
