"""Run the trajectory-tracking controller on the planar or 3D drone and plot the result.

Examples (from the repo root):
    python scripts/run_tracking.py --model planar --traj circle
    python scripts/run_tracking.py --model planar --traj figure8 --no-ff        # no feedforward
    python scripts/run_tracking.py --model 3d --traj figure8 --plot-file f8.png
    python scripts/run_tracking.py --model 3d --traj waypoints --video wp.mp4
    python scripts/run_tracking.py --model 3d --traj circle --no-rate-comp       # fails: see below

Trajectories: hover, circle, figure8, waypoints. Planar shapes are drawn in the y-z
plane, 3D shapes in the x-y plane. Positions are relative to the spawn point.

The 3D runs use drone_pendulum.xml, where the pendulum hangs from hinges about the body
x-axis (like the real hardware's bearings). --no-rate-comp turns off the controller's
inertia-aware rate-loop compensation to show why it is needed: without it the drone
cannot pitch fast enough with the rigidly attached payload and tracking along x fails.
"""

import argparse

import gymnasium as gym
import numpy as np

import drone_pendulum  # noqa: F401  (registers envs)
from drone_pendulum.controllers.trajectories import make_trajectory
from drone_pendulum.controllers.tracking_2d import PlanarTrackingController
from drone_pendulum.controllers.tracking_3d import TrackingController3D
from drone_pendulum.envs.planar_drone_pendulum_env import PlanarDronePendulumEnv, PlanarEnvConfig
from drone_pendulum.envs.registration import ENV_ID


def make_env_and_controller(args):
    traj_kwargs = {}
    if args.traj in ("circle",):
        traj_kwargs = dict(radius=args.size, period=args.period)
    elif args.traj == "figure8":
        traj_kwargs = dict(width=2 * args.size, height=args.size, period=args.period)
    traj = make_trajectory(args.traj, planar=args.model == "planar", **traj_kwargs)
    render_mode = "rgb_array" if args.video else None

    if args.model == "planar":
        env = PlanarDronePendulumEnv(PlanarEnvConfig(max_episode_seconds=args.seconds),
                                     render_mode=render_mode)
        ctrl = PlanarTrackingController(env, traj, use_feedforward=not args.no_ff)
    else:
        env = gym.make(ENV_ID, render_mode=render_mode, episode_limit=1)
        steps = int(round(args.seconds / env.unwrapped.control_dt))
        env.close()
        env = gym.make(ENV_ID, render_mode=render_mode, episode_limit=steps)
        ctrl = TrackingController3D(env, traj, use_feedforward=not args.no_ff,
                                    compensate_rate_loop=not args.no_rate_comp)
    return env, ctrl, traj


def drone_position(env, obs, model):
    if model == "planar":
        return np.r_[0.0, obs[0:2]]
    return np.asarray(obs["drone_position_delta"], dtype=float)


def link_angles(env, model):
    """Swing of each link from vertical, in degrees (world frame)."""
    u = env.unwrapped
    if model == "planar":
        return np.rad2deg(np.abs(u.sim.drone_model.get_absolute_link_angles()))
    data, m = u.simulator.data, u.simulator.model
    out = []
    for body in ("link1", "link2"):
        axis = data.xmat[m.body(body).id].reshape(3, 3)[:, 2]  # link points along -z_local
        out.append(np.rad2deg(np.arccos(np.clip(axis[2], -1.0, 1.0))))
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["planar", "3d"], default="planar")
    p.add_argument("--traj", choices=["hover", "circle", "figure8", "waypoints"], default="circle")
    p.add_argument("--size", type=float, default=0.5, help="circle radius / figure-8 height [m]")
    p.add_argument("--period", type=float, default=5.0, help="seconds per lap")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--no-ff", action="store_true", help="disable acceleration/jerk feedforward")
    p.add_argument("--no-rate-comp", action="store_true",
                   help="3D only: disable the inertia-aware rate-loop compensation")
    p.add_argument("--plot-file", help="save the plot instead of opening a window")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--video", help="save an MP4 of the run")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env, ctrl, traj = make_env_and_controller(args)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.video, fps=round(1 / ctrl.dt))

    ts, pos, ref, swing, actions = [], [], [], [], []
    obs, info = env.reset(seed=args.seed)
    ctrl.reset()
    terminated = truncated = False
    try:
        while not (terminated or truncated):
            u = ctrl.act(obs, info)
            obs, _, terminated, truncated, info = env.step(u)
            ts.append(ctrl.t)
            pos.append(drone_position(env, obs, args.model))
            ref.append(traj(ctrl.t).pos)
            swing.append(link_angles(env, args.model))
            actions.append(u)
            if writer is not None:
                writer.append_data(env.render())
    finally:
        if writer is not None:
            writer.close()
        env.close()

    t, pos, ref = np.array(ts), np.array(pos), np.array(ref)
    swing, actions = np.array(swing), np.array(actions)
    err = np.linalg.norm(pos - ref, axis=1)
    settled = err[len(err) // 4:]
    print(f"{args.model} / {args.traj}: steps={len(t)} terminated={terminated} "
          f"reason={info.get('termination_reason', info.get('failure_reason'))}")
    print(f"tracking error: RMS={np.sqrt(np.mean(settled**2)):.4f} m (after first 25%), "
          f"max={err.max():.4f} m")
    print(f"max link swing: {swing.max(axis=0).round(1)} deg")

    if args.no_plot:
        return
    import matplotlib
    if args.plot_file:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes_idx = (1, 2) if args.model == "planar" else (0, 1, 2)
    names = "xyz"
    fig = plt.figure(figsize=(12, 8))
    ax_path = fig.add_subplot(2, 2, 1)
    a, b = (1, 2) if args.model == "planar" else (0, 1)
    ax_path.plot(ref[:, a], ref[:, b], "k--", label="reference")
    ax_path.plot(pos[:, a], pos[:, b], label="drone")
    ax_path.set_xlabel(f"{names[a]} [m]"); ax_path.set_ylabel(f"{names[b]} [m]")
    ax_path.set_aspect("equal", adjustable="datalim"); ax_path.legend(); ax_path.set_title("path")

    ax_xyz = fig.add_subplot(2, 2, 2)
    for i in axes_idx:
        line, = ax_xyz.plot(t, pos[:, i], label=names[i])
        ax_xyz.plot(t, ref[:, i], "--", color=line.get_color())
    ax_xyz.set_ylabel("position [m] (dashed = ref)"); ax_xyz.legend()

    ax_err = fig.add_subplot(2, 2, 3)
    ax_err.plot(t, err * 100, label="tracking error [cm]")
    ax_err.plot(t, swing[:, 0], label="link 1 swing [deg]")
    ax_err.plot(t, swing[:, 1], label="link 2 swing [deg]")
    ax_err.set_xlabel("time [s]"); ax_err.legend()

    ax_u = fig.add_subplot(2, 2, 4)
    labels = ["thrust", "roll rate"] if args.model == "planar" else ["thrust", "roll rate", "pitch rate", "yaw rate"]
    for k, lab in enumerate(labels):
        ax_u.step(t, actions[:, k], label=lab)
    ax_u.set_xlabel("time [s]"); ax_u.set_ylabel("normalized action"); ax_u.legend()

    for ax in fig.axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{args.model} tracking: {args.traj}"
                 f"{' (no feedforward)' if args.no_ff else ''}")
    fig.tight_layout()
    if args.plot_file:
        fig.savefig(args.plot_file, dpi=110)
    else:
        plt.show()


if __name__ == "__main__":
    main()
