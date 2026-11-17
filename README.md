# DronePendulumRL

Reinforcement learning for stabilizing a double pendulum suspended from a drone.

## Setup

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. From the repository root, install the project:

   ```bash
   pip install -e .
   ```

This installs the project (and all it's dependencies) in editable mode, so changes to the source code are immediately reflected without reinstalling.

## Dependencies

Project dependencies are managed in `pyproject.toml`, under `[project]` → `dependencies`.

To add a dependency:
1. Add its package name to the `dependencies` list in `pyproject.toml`. Potentially specify version so we don't end up with conflicts.
2. Commit and push the updated file.
3. Other teammates should pull the changes and run:

   ```bash
   pip install -e .
   ```

Pip automatically installs any new dependencies and updates packages when necessary. No need to reinstall after ordinary source code changes.

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
| `controllers/lqr_2d.py` | LQR baseline | linearization + DARE are **TODO** |
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
