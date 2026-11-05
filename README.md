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