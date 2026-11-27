"""Labeled demo videos: same hard start, several m2 values, PPO with the PD settle time noted.

    DP_MODEL=runs/run3/model.xml uv run python demo_videos.py --run run3

Start: drone level and still at the target, link 1 at 58 deg (just inside where the
upper arm meets the landing gear), link 2 a further 35 deg (93 deg from vertical, above
horizontal), both at rest. Nominal physics, no sensor noise, 1-step latency.
"""

import argparse
import json
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import evaluate as ev
from dp.env import DronePendulumEnv

Q0 = np.radians([58.0, 35.0])
TRAIN_RANGE = (0.049, 0.150)
SECONDS = 12.0
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def hard_reset(env, m2):
    env.reset(seed=0, options={"m2": max(m2, 1e-6)})
    d = env.data
    d.qpos[:] = 0
    d.qpos[:3] = env.target
    d.qpos[3] = 1.0
    d.qpos[env.jq:env.jq + 2] = Q0
    d.qvel[:] = 0
    mujoco.mj_forward(env.model, d)
    env.rate_int = np.zeros(3)
    return env._obs()


def run(env, policy, m2, cam=None, renderer=None):
    obs = hard_reset(env, m2)
    frames, swings = [], []
    for _ in range(env.max_steps):
        obs, _, term, _, info = env.step(policy(obs))
        swings.append(info["swing_deg"].max())
        if renderer is not None:
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render())
        if term:
            break
    swings = np.array(swings)
    above = np.nonzero(swings > 5.0)[0]
    settle = None if term else ((above[-1] + 1) * 0.02 if len(above) else 0.0)
    return frames, swings, settle, term


def label(frames, swings, m2, settle, pd_settle, crashed):
    f_big, f_small, f_b = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT_B, 22)
    unseen = not (TRAIN_RANGE[0] - 1e-9 <= m2 <= TRAIN_RANGE[1] + 1e-9)
    title = f"m2 = {m2 * 1000:.0f} g" + ("   (UNSEEN IN TRAINING)" if unseen else "   (in training range 49-150 g)")
    fmt = lambda s: "crashed" if s is None else f"{s:.2f} s"
    out = []
    for i, fr in enumerate(frames):
        img = Image.fromarray(fr)
        dr = ImageDraw.Draw(img, "RGBA")
        dr.rectangle([0, 0, img.width, 92], fill=(0, 0, 0, 170))
        dr.text((18, 10), title, font=f_big, fill=(255, 170, 60) if unseen else (255, 255, 255))
        dr.text((18, 54), f"PPO settle (<5 deg): {fmt(settle)}     PD baseline settle: {fmt(pd_settle)}",
                font=f_small, fill=(220, 220, 220))
        t = (i + 1) * 0.02
        settled = settle is not None and t >= settle
        dr.rectangle([0, img.height - 44, img.width, img.height], fill=(0, 0, 0, 170))
        dr.text((18, img.height - 36), f"t = {t:5.2f} s    max link angle = {swings[i]:5.1f} deg",
                font=f_small, fill=(220, 220, 220))
        if settled:
            dr.text((img.width - 150, img.height - 36), "SETTLED", font=f_b, fill=(90, 230, 120))
        out.append(np.asarray(img))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="run3")
    p.add_argument("--m2", default="0,50,100,150,200", help="grams")
    args = p.parse_args()
    out = Path("runs") / args.run
    kw_path = out / "env_kwargs.json"
    env_kwargs = json.loads(kw_path.read_text()) if kw_path.exists() else {}

    model = PPO.load(str(out / "final"), device="cpu")
    norm = VecNormalize.load(str(out / "vecnormalize.pkl"), DummyVecEnv([DronePendulumEnv]))
    norm.training = False
    ppo = lambda o: model.predict(norm.normalize_obs(o), deterministic=True)[0]

    env = DronePendulumEnv(randomize=False, episode_seconds=SECONDS, **env_kwargs)
    env.model.site_rgba[env.model.site("target").id, 3] = 0.35
    renderer = mujoco.Renderer(env.model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = env.target + np.array([0, 0, -0.1])
    cam.azimuth, cam.elevation, cam.distance = 120.0, -15.0, 0.85

    vids = []
    for g in [float(x) for x in args.m2.split(",")]:
        m2 = g / 1000
        env.model.geom_rgba[env.model.geom("m2_geom").id, 3] = 0.0 if g == 0 else 1.0
        _, _, pd_settle, _ = run(env, lambda o: ev.hover_baseline(env), m2)
        frames, swings, settle, crashed = run(env, ppo, m2, cam, renderer)
        vid = label(frames, swings, m2, settle, pd_settle, crashed)
        path = out / f"demo_m2_{g:.0f}g.mp4"
        imageio.mimsave(path, vid, fps=50, quality=8)
        print(f"m2={g:.0f}g  PPO settle={settle}  PD settle={pd_settle}  -> {path}")
        vids.append(vid)


if __name__ == "__main__":
    main()
