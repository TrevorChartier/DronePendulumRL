# DronePendulumRL

Reinforcement learning for stabilizing a double pendulum suspended from a drone.

## Setup

Use Python 3.11. From the repository root, choose either setup:

With pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

With uv (creates the environment automatically):

```bash
uv sync --python 3.11
```

Both install the project in editable mode, so source changes take effect without reinstalling.

## Dependencies

Dependencies are listed in `pyproject.toml`. After dependency changes, run
`pip install -e .` in your activated environment, or `uv sync`.
If you change dependencies using pip, also refresh `uv.lock` with `uv lock`
before committing. The lockfile is used by uv; pip ignores it.

## Project Structure

See `docs/architecture.md`

## Planar (2D) model — stage 1

A planar version of the system (drone moves in the Y-Z plane, rolls about X; both links
swing in the same plane) for validating the training pipeline and reward before 3D.

| File | What it is | Status |
| --- | --- | --- |
| `simulation/assets/drone_pendulum_2d.xml` | MJCF (same masses/lengths as 3D) | done |
| `simulation/model_2d.py` | state accessors, 10-dim state vector | done |
| `simulation/actuation_2d.py` | `[thrust, roll_rate]` → wrench, P rate loop | done |
| `simulation/simulator_2d.py` | reset / step / `apply_disturbance` | done |
| `envs/planar_drone_pendulum_env.py` | Gymnasium env `PlanarDronePendulum-v0` | reward, termination, reset distribution, disturbance schedule are **TODO** |
| `controllers/lqr_2d.py` | LQR baseline | done |
| `controllers/cascaded_pd_2d.py` | payload-unaware baseline | **TODO** (superseded by `tracking_2d.py` with a `Hover()` reference) |
| `controllers/tracking_2d.py`, `tracking_3d.py` | trajectory tracking (planar / 3D) | done, see below |
| `evaluation/rollout.py`, `metrics.py` | rollouts / metrics | rollout done, metrics **TODO** |
| `scripts/run_planar.py` | run + view + plot any controller | done |
| `scripts/train_planar.py` | training entry point | **TODO** |

State vector: `x = [y, z, roll, q1, q2, y_dot, z_dot, roll_dot, q1_dot, q2_dot]`, with
`q1, q2` relative joint angles (absolute link angles: `phi1 = roll + q1`, `phi2 = phi1 + q2`).
A zero action is hover thrust.

```bash
python scripts/run_planar.py --render --push 3.0
```

```bash
python -m pytest -q
```

## Trajectory tracking (planar and 3D)

Payload-unaware, differential-flatness tracking controllers that output the envs' own
`[thrust, body rates]` actions. They use only the drone's state, so with a `Hover()`
reference they double as the proposal's payload-unaware baseline, and their output can
be summed with a residual RL policy for trajectory following.

| File | What it is |
| --- | --- |
| `controllers/trajectories.py` | `Hover`, `Circle`, `Figure8`, `MinJerkWaypoints`: pos/vel/acc/jerk/yaw vs. time, relative to spawn |
| `controllers/tracking_2d.py` | `PlanarTrackingController`: position PD + accel/jerk feedforward → thrust, roll, roll rate |
| `controllers/tracking_3d.py` | `TrackingController3D`: SE(3) geometric controller at body-rate level (Lee / Mellinger) |
| `scripts/run_tracking.py` | run either one, plot reference vs. actual, optional video |

```bash
python scripts/run_tracking.py --model planar --traj figure8
```

```bash
python scripts/run_tracking.py --model 3d --traj circle --plot-file circle.png
```

Typical RMS tracking error with the default trajectories: about 1 cm planar, under 1 cm
3D. Turning off the feedforward (`--no-ff`) makes it 10–20x worse.

Both controllers take `mass=` (and the 3D one `inertia=`) to set what they assume about
the vehicle, plus optional integral gains `ki=`. Passing `bare_drone_mass(env)` /
`bare_drone_params(model)` gives a "regular drone" controller tuned without the payload.
Even with an integrator, that one tracks 4–8x worse in planar and tips over in 3D on
every moving trajectory, because it underestimates the pitch inertia with the payload
attached.

**Single-axis hinges and the 3D rate loop.** As on the real drone (bearings, not a ball
joint), the pendulum joints in `drone_pendulum.xml` only rotate about the body x-axis.
The payload swings freely in roll but is rigidly attached in pitch, so the drone's
effective pitch inertia is about 11x its roll inertia (0.13 vs 0.012 kg m²). The env's
inner rate loop applies the same proportional gain on every axis, so plain body-rate
commands are tracked far too slowly in pitch and the drone can't move along x.
`TrackingController3D` therefore computes the effective inertia from the MuJoCo model
(`effective_rotational_inertia`) and scales its rate commands so the P loop produces
the torque the desired rate response needs (`--no-rate-comp` shows the failure without
it). On hardware, the flight controller's own rate loop replaces this P loop: turn the
compensation off, or set `rate_loop_gain` / `k_rate` to match it.

## Planar PPO training and evaluation

Run from the repository root after installing the project. With pip, activate your
virtual environment and run:

```bash
python scripts/train_hover.py --num-envs 8 --timesteps 1000000 --run-dir runs/hover_ppo
```

Use a new run directory for each training run. The policy and observation
normalization statistics are saved there.

Evaluate the trained policy:

```bash
python scripts/evaluate_hover.py runs/hover_ppo --episodes 10
```

Evaluate recovery from a 3 N push:

```bash
python scripts/evaluate_hover.py runs/hover_ppo --episodes 10 --push 3
```

Evaluation prints results/video and saves a JSON report in the run directory.
With uv, prefix these commands with `uv run`.
