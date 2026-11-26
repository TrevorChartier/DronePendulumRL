# DronePendulumRL

Model-based reinforcement learning for stabilizing a double pendulum suspended from a drone.

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

## Simulation & Environments

### Planar (2D) system

A planar version of the system (drone moves in the Y-Z plane, rolls about X; both links
swing in the same plane):

| File | What it is |
| --- | --- |
| `simulation/assets/drone_pendulum_2d.xml` | MJCF (same masses/lengths as 3D) |
| `simulation/model_2d.py` | State accessors, 10-dim state vector |
| `simulation/actuation_2d.py` | `[thrust, roll_rate]` -> wrench, P rate loop |
| `simulation/simulator_2d.py` | Reset / step / `apply_disturbance` |
| `envs/planar_drone_pendulum_env.py` | Base Gymnasium env `PlanarDronePendulum-v0` |
| `envs/hover_env.py` | Planar hover env with shaped reward/cost terms and safety limits |
| `controllers/lqr_2d.py` | Linearized discrete-time LQR baseline |
| `controllers/cascaded_pd_2d.py` | Payload-unaware cascaded PD baseline |
| `evaluation/rollout.py`, `metrics.py` | Generic controller trajectory rollouts and metrics |
| `scripts/run_planar.py` | Run, view, record, and plot any `Controller` on the planar system |

State vector: `x = [y, z, roll, q1, q2, y_dot, z_dot, roll_dot, q1_dot, q2_dot]`, with
`q1, q2` relative joint angles (absolute link angles: `phi1 = roll + q1`, `phi2 = phi1 + q2`).
A zero action corresponds to hover thrust.

```bash
python scripts/run_planar.py --controller lqr --render --push 3.0
```

### 3D systems

- `simulation/` + `envs/drone_pendulum_env.py` (`DronePendulum-v0`): Generic 3D quadrotor + double pendulum MJCF model, simulator, and Gymnasium environment. Visualize with `python scripts/view_model.py`.
- `starling_simulation/` + `envs/starling_pendulum_env.py` (`StarlingPendulum-v0`): Identified Starling 2 Max MuJoCo dynamics, rotor/motor-delay model, inner-loop local waypoint controller, and Gymnasium environment.

## Tests

```bash
python -m pytest -q
```
