
"""Basic tests for the drone-pendulum simulator."""

from pathlib import Path

import numpy as np

from drone_pendulum.simulation.simulator import DronePendulumSimulator


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "drone_pendulum"
    / "simulation"
    / "assets"
    / "drone_pendulum.xml"
)


def test_simulator_initialization():
    """Check that the simulator and its model initialize correctly."""
    simulator = DronePendulumSimulator(str(MODEL_PATH))

    assert simulator.model is not None
    assert simulator.data is not None
    assert simulator.drone_model.total_mass > 0


def test_state_retrieval():
    """Check that the simulator returns all expected state quantities."""
    simulator = DronePendulumSimulator(str(MODEL_PATH))

    state = simulator.get_state()

    expected_shapes = {
        "drone_position": (3,),
        "drone_orientation": (4,),
        "drone_linear_velocity": (3,),
        "drone_angular_velocity": (3,),
        "joint_positions": (2,),
        "joint_velocities": (2,),
    }

    assert state.keys() == expected_shapes.keys()

    for key, shape in expected_shapes.items():
        assert state[key].shape == shape
        assert np.all(np.isfinite(state[key]))


def test_simulation_step():
    """Check that physics advances when an action is applied."""
    simulator = DronePendulumSimulator(str(MODEL_PATH))

    initial_time = simulator.data.time

    action = np.zeros(4)
    n_substeps = 10

    simulator.step(action, n_substeps=n_substeps)

    expected_time = initial_time + (
        n_substeps * simulator.model.opt.timestep
    )

    assert np.isclose(simulator.data.time, expected_time)

    state = simulator.get_state()

    for value in state.values():
        assert np.all(np.isfinite(value))


def test_simulation_reset():
    """Check that reset returns the simulation to its initial state."""
    simulator = DronePendulumSimulator(str(MODEL_PATH))

    simulator.step(np.zeros(4), n_substeps=10)
    simulator.reset()

    assert simulator.data.time == 0.0
    assert np.allclose(simulator.data.qvel, 0.0)

    state = simulator.get_state()

    for value in state.values():
        assert np.all(np.isfinite(value))


def test_state_is_independent_of_simulator():
    """Check that modifying returned state does not modify MuJoCo state."""
    simulator = DronePendulumSimulator(str(MODEL_PATH))

    state = simulator.get_state()
    original_position = simulator.drone_model.get_drone_position()

    state["drone_position"][:] = 100.0

    assert np.allclose(
        simulator.drone_model.get_drone_position(),
        original_position,
    )
