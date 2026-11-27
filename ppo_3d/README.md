# PPO stabilization of the 3D drone + double pendulum

Standalone MuJoCo + Stable-Baselines3 setup that trains a PPO policy to damp the swing
of a passive double pendulum hanging under the Starling 2 Max while holding position.
Self-contained: it does not import the rest of this repo.

## Model (`assets/drone_pendulum.xml`)

- Geometry, COMs and inertias from Percy's SolidWorks export (`mbrl` branch,
  `urdf/drone_pendulum_description`).
- Drone: 0.6 kg CAD body (FLU, x along the long motor axis). Rotors placed from the mesh
  at x = ±0.13 m, y = ±0.095 m, z = 0.0195 m.
  Note: `starling2max_pendulum.xml` on `mbrl` has its rotor sites rotated 90° relative to this CAD frame.
- Link 1: 164.6 g (CAD "Upper Arm Assy", assumed to include m1 = 79 g). Link 2: 67 g
  ("Lower Arm Assy"). m2: point mass at the link-2 tip, varied 49–150 g in training.
- Both hinges along body +Y (pendulum swings in the body XZ plane).

## Environment (`dp/env.py`)

Designed for sim-to-real through PX4 offboard attitude-rate setpoints:

- **Action** (4, in [-1, 1]): collective thrust and body rates.
  `action_to_setpoint` maps it to `thrust_norm` in [0, 1] and body rates (FLU, ±6/6/2 rad/s).
  For PX4 FRD send `p, -q, -r`.
- **Observation** (32): position error to target, velocity (world ENU), body rotation
  matrix, gyro (body FLU), sin/cos of both joint angles, joint rates, last two actions.
  `build_obs` is the layout to reproduce on hardware (mocap for the drone and links).
- **Sim pipeline:** latency queue (0–2 steps) → body-rate PI loop (PX4-like) → mixer →
  Starling D0012 rotor model (2000–12000 RPM, kT = 3.96e-6, τ up/down 55/85 ms) → MuJoCo.
  Attitude is fully physical, so the pendulum torques the drone.
- **Domain randomization:** m2, body masses ±10 %, per-rotor thrust scale (0.9–1.08),
  motor time constants, rate-loop gains ±30 %, joint damping, latency, sensor noise.
- **Reward:** alive bonus minus position error, velocity, link angle from world vertical
  (main term), link swing rate, body rates, tilt and action changes. Episodes are 10 s at 50 Hz.
  A crash (tilt past 90° or more than 2.5 m from target) gives −50.

## Usage

```bash
cd ppo_3d
uv sync
# train (≈1 h on 32 cores)
OMP_NUM_THREADS=1 PYTHONPATH=. uv run python train.py --steps 30000000 --envs 32 --name myrun
# evaluate vs a pendulum-unaware hover PD baseline (+ videos)
DP_MODEL=runs/run3/model.xml PYTHONPATH=. uv run python evaluate.py --run run3 --video
# labeled hard-start demo videos at several m2
DP_MODEL=runs/run3/model.xml MUJOCO_GL=egl PYTHONPATH=. uv run python demo_videos.py --run run3
# export a numpy-only policy
DP_MODEL=runs/run3/model.xml PYTHONPATH=. uv run python export_policy.py --run run3
```

## Trained policy (`runs/run3/`)

- `final.zip` + `vecnormalize.pkl`: SB3 model and observation normalization.
- `policy.npz`: numpy-only weights and normalization. Use `NumpyPolicy` in `export_policy.py`;
  it matches SB3 to about 2e-7.
- `model.xml` / `env_kwargs.json`: the exact model and env config it was trained with.

Results: 30M steps, 90 randomized test episodes. Settle time is how long until both links stay within 5° of vertical.

| | PPO | Hover-only PD |
|---|---|---|
| Median settle time | 2.3–2.5 s | 4.3–5.3 s |
| Residual swing (last 2 s) | 1.1–1.4° | 4.6–7.2° |
| Position hold error | 8–9 cm | 13–15 cm |
| Crashes | 1 / 90 | 1 / 90 |

Hard start (link 1 at 58° and link 2 at 93° from vertical, released from rest): PPO settles
in about 3 s for m2 = 0–200 g (0 and 200 g were never seen in training), while the PD takes 4.3–7.7 s.

## Known gaps

- The drone mass is a round 0.6 kg CAD override. Weigh the real vehicle.
- Heading isn't penalized, so the policy yaws freely while damping.
- Joint angles are assumed to be measured (mocap planned).
