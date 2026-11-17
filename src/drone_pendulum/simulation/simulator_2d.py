"""High-level simulator for the planar drone + double pendulum.

Responsibilities (same split as simulator.py for 3D):
    - load the model and the actuation layer
    - reset to a given state
    - apply an action and advance physics
    - apply external disturbances (pushes) for disturbance-recovery experiments
    - report the physical state
"""

from pathlib import Path

import mujoco
import numpy as np

from drone_pendulum.simulation.actuation_2d import PlanarActuation
from drone_pendulum.simulation.model_2d import STATE_DIM, PlanarDronePendulumModel

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "assets" / "drone_pendulum_2d.xml")


class PlanarDronePendulumSimulator:
    """Owns the MuJoCo model/data and steps the planar system."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, **actuation_kwargs) -> None:
        self.drone_model = PlanarDronePendulumModel(model_path)
        self.actuation = PlanarActuation(self.drone_model, **actuation_kwargs)

        self.model = self.drone_model.model
        self.data = self.drone_model.data

        # Active disturbance: {body_id: (force_yz, steps_remaining)}
        self._disturbances: dict[int, tuple[np.ndarray, int]] = {}

    @property
    def dt(self) -> float:
        """Physics timestep in seconds."""
        return float(self.model.opt.timestep)

    def reset(self, state: np.ndarray | None = None) -> None:
        """Reset to the default hover pose, or to `state` (10-vector) if given."""
        mujoco.mj_resetData(self.model, self.data)
        self._disturbances.clear()
        if state is not None:
            if np.shape(state) != (STATE_DIM,):
                raise ValueError(f"state must have shape ({STATE_DIM},)")
            self.drone_model.set_state_vector(state)
        else:
            mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray, n_substeps: int = 1) -> None:
        """Hold `action` constant for n_substeps physics steps (zero-order hold)."""
        if n_substeps < 1:
            raise ValueError("n_substeps must be at least 1.")
        for _ in range(n_substeps):
            self.actuation.apply_action(action)
            self._apply_disturbances()
            mujoco.mj_step(self.model, self.data)

    def apply_disturbance(self, body: str, force_yz, duration: float) -> None:
        """
        Push a body with a constant world-frame force for `duration` seconds.

        Example: a 0.1 s, 2 N sideways shove on the bottom mass
            sim.apply_disturbance("link2", force_yz=[2.0, 0.0], duration=0.1)

        Parameters
        ----------
        body : str
            "drone", "link1" or "link2".
        force_yz : array-like, shape (2,)
            Force in Newtons along world (y, z).
        duration : float
            Seconds. Impulse delivered = force * duration.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            raise ValueError(f"Unknown body '{body}'")
        n_steps = max(1, int(round(duration / self.dt)))
        self._disturbances[body_id] = (np.asarray(force_yz, dtype=float), n_steps)

    def _apply_disturbances(self) -> None:
        # The actuation layer overwrites the drone's row every step, so we ADD on top of it.
        # Pendulum link rows are only touched here, so clear them first.
        self.data.xfrc_applied[self.drone_model.link1_id] = 0.0
        self.data.xfrc_applied[self.drone_model.link2_id] = 0.0

        for body_id in list(self._disturbances):
            force_yz, steps_left = self._disturbances[body_id]
            self.data.xfrc_applied[body_id, 1:3] += force_yz
            if steps_left <= 1:
                del self._disturbances[body_id]
            else:
                self._disturbances[body_id] = (force_yz, steps_left - 1)

    def get_state(self) -> dict[str, np.ndarray]:
        """Physical state as a dict of named arrays (copies, safe to modify)."""
        m = self.drone_model
        return {
            "state_vector": m.get_state_vector(),
            "drone_position": m.get_drone_position(),
            "drone_velocity": m.get_drone_velocity(),
            "drone_roll": np.array([m.get_drone_roll()]),
            "drone_roll_rate": np.array([m.get_drone_roll_rate()]),
            "joint_positions": m.get_joint_positions(),
            "joint_velocities": m.get_joint_velocities(),
            "absolute_link_angles": m.get_absolute_link_angles(),
            "absolute_link_rates": m.get_absolute_link_rates(),
        }
