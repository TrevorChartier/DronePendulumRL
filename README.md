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
| `controllers/cascaded_pd_2d.py` | payload-unaware baseline | **TODO** |
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
