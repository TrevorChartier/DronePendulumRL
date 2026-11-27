"""Export a trained policy to a dependency-light .npz (numpy only) for the Starling.

    uv run python export_policy.py --run run1

Writes runs/<run>/policy.npz with the MLP weights and observation normalization, and
checks that numpy inference matches SB3. On the drone, use `NumpyPolicy` below
(copy this file) with `DronePendulumEnv.build_obs` / `action_to_setpoint` semantics:

    obs = build_obs(pos - target, vel_enu, R_body_to_world, gyro_flu, q, qd, last_two_actions)
    a = policy(obs)
    thrust_norm = clip(hover_thrust_norm * (1 + a[0]), 0, 1)
    rates_flu = a[1:] * max_rate          # PX4 FRD: p, -q, -r
"""

import argparse
from pathlib import Path

import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        z = np.load(path)
        self.W = [z[f"W{i}"] for i in range(int(z["n_layers"]))]
        self.b = [z[f"b{i}"] for i in range(int(z["n_layers"]))]
        self.mean, self.var, self.clip = z["obs_mean"], z["obs_var"], float(z["clip_obs"])
        self.hover_thrust_norm = float(z["hover_thrust_norm"])
        self.max_rate = z["max_rate"]

    def __call__(self, obs):
        x = np.clip((obs - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = W @ x + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)


def main():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from dp.env import MAX_RATE, DronePendulumEnv

    p = argparse.ArgumentParser()
    p.add_argument("--run", default="run1")
    p.add_argument("--ckpt", default="final")
    args = p.parse_args()
    out = Path("runs") / args.run
    model = PPO.load(str(out / args.ckpt), device="cpu")
    norm = VecNormalize.load(str(out / "vecnormalize.pkl"), DummyVecEnv([DronePendulumEnv]))

    linears = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    linears.append(model.policy.action_net)
    arrays = {f"W{i}": l.weight.detach().numpy() for i, l in enumerate(linears)}
    arrays.update({f"b{i}": l.bias.detach().numpy() for i, l in enumerate(linears)})
    env = DronePendulumEnv()
    np.savez(
        out / "policy.npz", n_layers=len(linears), obs_mean=norm.obs_rms.mean, obs_var=norm.obs_rms.var,
        clip_obs=norm.clip_obs, hover_thrust_norm=env.hover_thrust_norm, max_rate=MAX_RATE, **arrays,
    )

    pol = NumpyPolicy(out / "policy.npz")
    obs, _ = env.reset(seed=0)
    err = 0.0
    for _ in range(200):
        a_np = pol(obs)
        a_sb = model.predict(norm.normalize_obs(obs), deterministic=True)[0]
        err = max(err, np.abs(a_np - np.clip(a_sb, -1, 1)).max())
        obs, *_ = env.step(a_np)
    print(f"wrote {out / 'policy.npz'}; max |numpy - sb3| = {err:.2e}")


if __name__ == "__main__":
    main()
