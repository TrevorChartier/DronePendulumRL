"""Run every experiment used in the progress report; write figures/ and results.json.

Run from the report/ folder with the project installed, e.g.
    python scripts/make_report_results.py
then update the numbers in main.tex from results.json and rebuild the PDF.
"""

import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import drone_pendulum  # noqa: F401
from drone_pendulum.controllers.base import ZeroController
from drone_pendulum.controllers.lqr_2d import LQRController
from drone_pendulum.controllers.trajectories import Hover, make_trajectory
from drone_pendulum.controllers.tracking_2d import PlanarTrackingController, bare_drone_mass
from drone_pendulum.controllers.tracking_3d import (
    TrackingController3D, bare_drone_params, effective_rotational_inertia)
from drone_pendulum.envs.planar_drone_pendulum_env import PlanarDronePendulumEnv, PlanarEnvConfig
from drone_pendulum.envs.registration import ENV_ID

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.1})
ANG_TOL, POS_TOL = np.deg2rad(2.0), 0.05
results = {}


def planar_env(seconds):
    return PlanarDronePendulumEnv(PlanarEnvConfig(
        max_episode_seconds=seconds, init_joint_angle_range=0.0, init_joint_vel_range=0.0))


def settling_time(t, x, t0):
    """First time after t0 from which links stay within 2 deg and drone within 5 cm."""
    phi1 = x[:, 2] + x[:, 3]
    phi2 = phi1 + x[:, 4]
    ok = (np.abs(phi1) < ANG_TOL) & (np.abs(phi2) < ANG_TOL) & (np.hypot(x[:, 0], x[:, 1]) < POS_TOL)
    bad = np.where(~ok)[0]
    if len(bad) == 0:
        return 0.0
    if bad[-1] == len(ok) - 1:
        return float("inf")
    return float(t[bad[-1] + 1] - t0)


def planar_rollout(env, ctrl, x0=None, push=None, seconds=None):
    obs, info = env.reset(seed=0, options={"state": x0} if x0 is not None else None)
    ctrl.reset()
    ts, xs, us = [info["time"]], [info["state"]], []
    pushed = push is None
    while True:
        if not pushed and info["time"] >= push[0]:
            env.sim.apply_disturbance("link2", [push[1], 0.0], 0.1)
            pushed = True
        u = ctrl.act(obs, info)
        obs, _, term, trunc, info = env.step(u)
        ts.append(info["time"]); xs.append(info["state"]); us.append(u)
        x = info["state"]
        if not np.all(np.isfinite(x)) or np.hypot(x[0], x[1]) > 2.0:
            return np.array(ts), np.array(xs), np.array(us), False
        if term or trunc:
            break
    return np.array(ts), np.array(xs), np.array(us), True


# ============================================================ E1: planar push recovery
env = planar_env(10.0)
controllers = {
    "LQR": LQRController.from_env(env),
    "Payload-unaware PD (hover)": PlanarTrackingController(env, Hover()),
    "Open loop (hover thrust)": ZeroController(2),
}
pushes = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
e1 = {}
traces = {}
for name, ctrl in controllers.items():
    rows = []
    for F in pushes:
        t, x, u, alive = planar_rollout(env, ctrl, x0=np.zeros(10), push=(1.0, F))
        phi = np.c_[x[:, 2] + x[:, 3], x[:, 2] + x[:, 3] + x[:, 4]]
        ts = settling_time(t, x, 1.0) if alive else float("inf")
        rows.append({
            "push_N": F, "survived": bool(alive), "settling_s": ts,
            "max_swing_deg": float(np.rad2deg(np.abs(phi).max())),
            "max_pos_err_m": float(np.hypot(x[:, 0], x[:, 1]).max()),
            "rms_pos_err_m": float(np.sqrt(np.mean(x[:, 0] ** 2 + x[:, 1] ** 2))),
            # largest link angle during the last 2 s of the 10 s episode
            "residual_swing_deg": float(np.rad2deg(np.abs(phi[t >= t[-1] - 2.0]).max())) if alive else None,
        })
        if F == 3.0:
            traces[name] = (t, x)
    e1[name] = rows
results["push_recovery"] = e1

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1))
for name, (t, x) in traces.items():
    phi2 = np.rad2deg(x[:, 2] + x[:, 3] + x[:, 4])
    axes[0].plot(t, phi2, label=name)
    axes[1].plot(t, x[:, 0] * 100, label=name)
axes[0].set_ylabel(r"lower link angle $\phi_2$ [deg]")
axes[1].set_ylabel("drone lateral offset $y$ [cm]")
for ax in axes:
    ax.set_xlabel("time [s]"); ax.grid(alpha=0.3); ax.axvline(1.0, color="k", lw=0.6, ls=":")
axes[0].set_ylim(-65, 65)
axes[1].set_ylim(-45, 45)
axes[1].legend(loc="lower right", frameon=False, fontsize=6)
fig.subplots_adjust(wspace=0.32)
fig.savefig(FIG / "push_recovery.pdf")
plt.close(fig)

# ============================================================ E2: initial-angle range
e2 = {}
for name in ("LQR", "Payload-unaware PD (hover)"):
    ctrl = controllers[name]
    best = 0
    for q1 in range(10, 181, 10):
        x0 = np.zeros(10); x0[3] = np.deg2rad(q1)
        t, x, u, alive = planar_rollout(env, ctrl, x0=x0)
        if alive and settling_time(t, x, 0.0) < np.inf:
            best = q1
        else:
            break
    e2[name] = best
results["max_initial_link_angle_deg"] = e2
env.close()

# ============================================================ E3: trajectory tracking
def track_planar(traj, seconds=12.0, **kw):
    env = planar_env(seconds)
    ctrl = PlanarTrackingController(env, traj, **kw)
    obs, info = env.reset(seed=0); ctrl.reset()
    T, P, Rf, S = [], [], [], []
    while True:
        obs, _, term, trunc, info = env.step(ctrl.act(obs, info))
        T.append(ctrl.t); P.append(np.r_[0.0, obs[0:2]]); Rf.append(traj(ctrl.t).pos)
        S.append(np.rad2deg(np.abs(info["absolute_link_angles"]).max()))
        if term or trunc or np.linalg.norm(P[-1] - Rf[-1]) > 2.0:
            break
    env.close()
    return np.array(T), np.array(P), np.array(Rf), np.array(S), not (term or np.linalg.norm(P[-1] - Rf[-1]) > 2.0)


def link_swing_3d(sim):
    out = []
    for b in ("link1", "link2"):
        z = sim.data.xmat[sim.model.body(b).id].reshape(3, 3)[:, 2]
        out.append(np.rad2deg(np.arccos(np.clip(z[2], -1, 1))))
    return max(out)


def track_3d(traj, seconds=12.0, **kw):
    env = gym.make(ENV_ID, episode_limit=int(round(seconds / 0.02)))
    ctrl = TrackingController3D(env, traj, **kw)
    obs, info = env.reset(seed=0); ctrl.reset()
    T, P, Rf, S = [], [], [], []
    term = False
    while True:
        obs, _, term, trunc, info = env.step(ctrl.act(obs, info))
        T.append(ctrl.t); P.append(np.asarray(obs["drone_position_delta"], float))
        Rf.append(traj(ctrl.t).pos); S.append(link_swing_3d(env.unwrapped.simulator))
        if term or trunc:
            break
    env.close()
    return np.array(T), np.array(P), np.array(Rf), np.array(S), not term


def summarize(T, P, Rf, S, ok):
    e = np.linalg.norm(P - Rf, axis=1)
    k = np.searchsorted(T, 3.0)
    return {"completed": bool(ok), "duration_s": float(T[-1]),
            "rms_err_cm": float(np.sqrt(np.mean(e[k:] ** 2)) * 100) if ok else None,
            "max_err_cm": float(e.max() * 100), "max_swing_deg": float(S.max())}


TRAJ_KW = {"circle": dict(radius=0.5, period=5.0), "figure8": dict(width=1.0, height=0.5, period=5.0),
           "waypoints": {}}
e3 = {"planar": {}, "3d": {}}
paths = {}
for name, kw in TRAJ_KW.items():
    tp = make_trajectory(name, planar=True, **kw)
    t3 = make_trajectory(name, planar=False, **kw)
    out = track_planar(tp); e3["planar"].setdefault("full", {})[name] = summarize(*out)
    if name == "figure8":
        paths["planar"] = out
    e3["planar"].setdefault("no_ff", {})[name] = summarize(*track_planar(tp, use_feedforward=False))
    out = track_3d(t3); e3["3d"].setdefault("full", {})[name] = summarize(*out)
    if name == "figure8":
        paths["3d"] = out
    e3["3d"].setdefault("no_ff", {})[name] = summarize(*track_3d(t3, use_feedforward=False))
    e3["3d"].setdefault("no_rate_comp", {})[name] = summarize(*track_3d(t3, compensate_rate_loop=False))
results["tracking"] = e3

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.0), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
T, P, Rf, S, _ = paths["planar"]
axes[0].plot(Rf[:, 1], Rf[:, 2], "k--", lw=0.8, label="reference")
axes[0].plot(P[:, 1], P[:, 2], label="drone")
axes[0].set_xlabel("$y$ [m]"); axes[0].set_ylabel("$z$ [m]"); axes[0].set_title("planar (y-z plane)")
axes[0].set_aspect("equal", adjustable="datalim"); axes[0].legend(frameon=False, loc="lower center")
T3, P3, R3, S3, _ = paths["3d"]
axes[1].plot(R3[:, 0], R3[:, 1], "k--", lw=0.8)
axes[1].plot(P3[:, 0], P3[:, 1])
axes[1].set_xlabel("$x$ [m]"); axes[1].set_ylabel("$y$ [m]"); axes[1].set_title("3D (x-y plane)")
axes[1].set_aspect("equal", adjustable="datalim")
axes[2].plot(T, np.linalg.norm(P - Rf, axis=1) * 100, label="planar error [cm]")
axes[2].plot(T3, np.linalg.norm(P3 - R3, axis=1) * 100, label="3D error [cm]")
axes[2].plot(T3, S3, lw=0.8, alpha=0.7, label="3D max link swing [deg]")
axes[2].set_xlabel("time [s]"); axes[2].set_ylim(0, 48); axes[2].legend(frameon=False, loc="upper left", ncol=1, fontsize=6)
for ax in axes:
    ax.grid(alpha=0.3)
fig.subplots_adjust(wspace=0.42)
fig.savefig(FIG / "tracking_figure8.pdf")
plt.close(fig)

# ============================================================ E4: payload-aware vs regular
J_eff = None
e4 = {"planar": {}, "3d": {}}
err_traces = {}
for name, kw in TRAJ_KW.items():
    tp = make_trajectory(name, planar=True, **kw)
    t3 = make_trajectory(name, planar=False, **kw)
    penv = planar_env(1.0); mb = bare_drone_mass(penv); penv.close()
    a = track_planar(tp, ki=(1.0, 3.0)); b = track_planar(tp, ki=(1.0, 3.0), mass=mb)
    e4["planar"][name] = {"aware": summarize(*a), "regular": summarize(*b)}
    genv = gym.make(ENV_ID, episode_limit=1)
    m, J = bare_drone_params(genv.unwrapped.simulator.model)
    J_eff = effective_rotational_inertia(genv.unwrapped.simulator.model)
    genv.close()
    c = track_3d(t3, ki=(1.0, 1.0, 3.0)); d = track_3d(t3, ki=(1.0, 1.0, 3.0), mass=m, inertia=J)
    e4["3d"][name] = {"aware": summarize(*c), "regular": summarize(*d)}
    if name == "figure8":
        err_traces = {"planar": (a, b), "3d": (c, d)}
results["aware_vs_regular"] = e4
results["inertia"] = {"effective_diag": np.diag(J_eff).tolist(), "bare_diag": np.diag(J).tolist(),
                      "bare_mass": m, "total_mass": None}

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.0))
for ax, key in zip(axes, ("planar", "3d")):
    (Ta, Pa, Ra, _, _), (Tb, Pb, Rb, _, okb) = err_traces[key]
    ax.plot(Ta, np.linalg.norm(Pa - Ra, axis=1) * 100, label="payload-aware")
    ax.plot(Tb, np.linalg.norm(Pb - Rb, axis=1) * 100, label="regular (bare-drone model)")
    if not okb:
        ax.plot(Tb[-1], np.linalg.norm(Pb[-1] - Rb[-1]) * 100, "rx", ms=6)
        ax.annotate("tilt limit", (Tb[-1], np.linalg.norm(Pb[-1] - Rb[-1]) * 100),
                    textcoords="offset points", xytext=(-38, -4), fontsize=7, color="r")
    ax.set_yscale("log"); ax.set_ylim(0.05, 300)
    ax.set_title(f"{'planar' if key == 'planar' else '3D'} figure-8")
    ax.set_xlabel("time [s]"); ax.set_ylabel("tracking error [cm]"); ax.grid(alpha=0.3, which="both")
axes[0].legend(frameon=False, loc="upper right")
fig.subplots_adjust(wspace=0.32)
fig.savefig(FIG / "aware_vs_regular.pdf")
plt.close(fig)

# ============================================================ snapshots of both models
genv = gym.make(ENV_ID, episode_limit=1)
sim = genv.unwrapped.simulator
sim.reset(joint_positions=np.array([0.5, -0.4]), joint_velocities=np.zeros(2))
r = mujoco.Renderer(sim.model, 480, 560)
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0, 0, 1.55]; cam.distance, cam.azimuth, cam.elevation = 2.6, 135, -15
r.update_scene(sim.data, camera=cam); img3 = r.render().copy(); r.close(); genv.close()
penv = planar_env(1.0)
penv.reset(options={"state": np.r_[0, 0, 0.1, 0.5, -0.4, np.zeros(5)]})
r = mujoco.Renderer(penv.sim.model, 480, 560)
cam2 = mujoco.MjvCamera(); cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
cam2.lookat[:] = [0, 0, 1.55]; cam2.distance, cam2.azimuth, cam2.elevation = 2.6, 180, 0
r.update_scene(penv.sim.data, camera=cam2); img2 = r.render().copy(); r.close()
results["inertia"]["total_mass"] = penv.sim.drone_model.total_mass
penv.close()
fig, axes = plt.subplots(1, 2, figsize=(4.2, 1.9))
for ax, img, title in zip(axes, (img2, img3), ("planar model", "3D model")):
    ax.imshow(img); ax.set_title(title); ax.axis("off")
fig.savefig(FIG / "models.png", dpi=250)
plt.close(fig)


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return "inf"
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


(ROOT / "results.json").write_text(json.dumps(_clean(results), indent=2))
print(json.dumps(_clean(results), indent=1))
