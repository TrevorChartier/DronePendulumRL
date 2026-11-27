"""Evaluate a trained policy against a pendulum-unaware hover baseline.

    uv run python evaluate.py --run run1 [--video]

Baseline: a cascaded PD position/attitude controller that only tries to hover at
the target (ignores the pendulum). Both see the same initial conditions.
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dp.env import F_MAX, MAX_RATE, DronePendulumEnv


def hover_baseline(env):
    d = env.data
    e = d.qpos[:3] - env.target
    v = d.qvel[:3]
    acc = -4.0 * e - 3.0 * v + np.array([0, 0, 9.81])
    R = d.xmat[env.drone_id].reshape(3, 3)
    z_des = acc / np.linalg.norm(acc)
    T = float(env.model.body_mass[env.drone_id:].sum()) * acc @ R[:, 2]
    err_body = R.T @ np.cross(R[:, 2], z_des)
    w_des = np.array([8.0 * err_body[0], 8.0 * err_body[1], 0.0])
    a0 = T / (4 * F_MAX) / env.hover_thrust_norm - 1.0
    return np.concatenate([[a0], np.clip(w_des / MAX_RATE, -1, 1)])


def rollout(env, policy, seed, frames=None, m2=None):
    obs, _ = env.reset(seed=seed, options=None if m2 is None else {"m2": m2})
    swings, errs, done = [], [], False
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        swings.append(info["swing_deg"].max())
        errs.append(info["pos_err"])
        if frames is not None:
            frames.append(env.render())
        done = term or trunc
    swings = np.array(swings)
    # settle time: first time after which the max link angle stays below 5 deg
    above = np.nonzero(swings > 5.0)[0]
    settle = (above[-1] + 1) * 0.02 if len(above) else 0.0
    if term:
        settle = np.inf
    return dict(crashed=bool(term), settle=settle, swing_last2s=swings[-100:].mean(),
                pos_err_mean=float(np.mean(errs)), swing0=swings[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="run1")
    p.add_argument("--ckpt", default="final")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--video", action="store_true")
    args = p.parse_args()
    out = Path("runs") / args.run

    model = PPO.load(str(out / args.ckpt), device="cpu")
    norm = VecNormalize.load(str(out / "vecnormalize.pkl"), DummyVecEnv([DronePendulumEnv]))
    norm.training = False
    ppo = lambda o: model.predict(norm.normalize_obs(o), deterministic=True)[0]

    import json
    kw_path = out / "env_kwargs.json"
    env_kwargs = json.loads(kw_path.read_text()) if kw_path.exists() else {}
    env = DronePendulumEnv(randomize=True, **env_kwargs)
    results = {}
    for name, pol in [("ppo", ppo), ("hover_pd", lambda o: hover_baseline(env))]:
      for m2 in (0.049, 0.10, 0.15):
        rs = [rollout(env, pol, 1000 + i, m2=m2) for i in range(args.episodes)]
        ok = [r for r in rs if not r["crashed"]] or rs
        results[name, m2] = rs
        print(f"{name:9s} m2={m2*1000:3.0f}g crashes {sum(r['crashed'] for r in rs):2d}/{len(rs)}  "
              f"median settle(<5deg) {np.median([r['settle'] for r in rs]):6.2f}s  "
              f"swing last 2s {np.mean([r['swing_last2s'] for r in ok]):5.2f}deg  "
              f"pos err {np.mean([r['pos_err_mean'] for r in ok]):.3f}m")

    if args.video:
        import imageio
        for name, pol in [("ppo", ppo), ("hover_pd", lambda o: hover_baseline(env))]:
            frames = []
            for s in (7, 8, 9):
                rollout(env, pol, s, frames)
            path = out / f"{name}.mp4"
            imageio.mimsave(path, frames, fps=50)
            print("wrote", path)


if __name__ == "__main__":
    main()
