# DronePendulumRL — System Architecture

## 1. Repository Structure

```text
TEMP - Just an idea, this could change/evolve as we go

DronePendulumRL/
├── README.md
├── architecture.md
├── pyproject.toml
│
├── configs/
│   ├── environment.yaml
│   ├── training.yaml
│   └── hardware.yaml
│
├── src/
│   └── drone_pendulum/
│       ├── __init__.py
│       │
│       ├── simulation/
│       │   ├── __init__.py
│       │   └── dynamics.py
│       │
│       ├── envs/
│       │   ├── __init__.py
│       │   └── drone_pendulum_env.py
│       │
│       ├── controllers/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── pid.py
│       │   ├── lqr.py
│       │   └── rl_controller.py
│       │
│       ├── rl/
│       │   ├── __init__.py
│       │   ├── policies.py
│       │   └── rewards.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── visualization.py
│       │
│       └── deployment/
│           ├── __init__.py
│           ├── drone_interface.py
│           └── state_estimation.py
│
├── scripts/
└── tests/
```

## 2. Core Components

### Simulation (`simulation/`)

Interfaces with MuJoCo.
- model.py: Loads and configures the MJCF model.
- simulator.py: Advances simulation, retrieves state, applies control inputs.
- assets/: Drone and double-pendulum MJCF models.

### Environment (`envs/`)

Provides the interface between the simulator and reinforcement learning algorithms.

The environment will follow the Gymnasium API and define:

- **Observations:** State information available to the controller/policy.
- **Actions:** Control commands provided by the policy.
- **Rewards:** Measures of stabilization performance.
- **Termination:** Conditions for ending an episode.
- **Reset:** Initial conditions for each episode.

The environment is responsible for translating policy actions into simulator commands and returning observations and rewards.

### Controllers (`controllers/`)

Contains implementations of different control strategies through a common controller interface.

Potential implementations include:

- PID controllers.
- LQR controllers.
- RL-based controllers.

Controllers should expose a consistent interface for receiving observations and producing control actions, allowing different approaches to be evaluated and deployed interchangeably.

An RL-based controller encapsulates a trained policy along with any required observation preprocessing, inference, and action postprocessing.

### Reinforcement Learning (`rl/`)

Contains components specific to reinforcement learning and policy learning.

Responsibilities include:

- Policy and value network architectures.
- Reward functions.
- Algorithm-specific components or integrations.
- Auxiliary learning components.

The RL implementation should interact with the environment through its standardized interface rather than directly accessing simulation internals.

Trained policy parameters can be loaded into an RL-based controller for evaluation or deployment.

### Training (`training/`)

Responsible for training and experiment execution. Defines how learning experiments are executed and supports reuse across different RL algorithms. This would contain something like the RL_Trainer class in the hw assignments. Perhaps this could also be changed to just a general infrastructure package.

Includes:

- Training loops and algorithm configuration.
- Checkpoint management.
- Logging and experiment tracking.
- Random seed management.

Training configurations should be externalized in `configs/` to support reproducible experiments.

Training produces trained policies that can subsequently be loaded into RL-based controllers.

### Evaluation (`evaluation/`)

Provides tools for measuring and visualizing controller performance.

Potential metrics include:

- Pendulum angular displacement.
- Maximum swing angle.
- Settling time.
- Drone position and orientation error.
- Control effort.
- Robustness to disturbances and changes in physical parameters.

Evaluation should support comparisons between learned policies and conventional controllers under consistent conditions.

### Deployment (`deployment/`)

Contains the interfaces required to run controllers on physical hardware.

Responsibilities include:

- Hardware communication.
- State estimation and observation processing.
- Controller initialization and execution.
- Translation of controller actions into hardware commands.
- Runtime safety constraints.

Deployment should be independent of the training pipeline and RL implementation. Any controller implementing the common controller interface should be usable.