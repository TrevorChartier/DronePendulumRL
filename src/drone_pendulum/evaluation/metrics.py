"""Evaluation metrics from the proposal. All take a Trajectory (see rollout.py).

TODO(you): implement these. Decide on (and write down!) precise definitions, e.g. what
"settled" means, so that every controller is scored the same way.
"""

import numpy as np

from drone_pendulum.evaluation.rollout import Trajectory


def absolute_link_angles(traj: Trajectory) -> np.ndarray:
    """(T, 2) world-frame link angles phi1 = roll + q1, phi2 = roll + q1 + q2."""
    roll, q1, q2 = traj.states[:, 2], traj.states[:, 3], traj.states[:, 4]
    return np.stack([roll + q1, roll + q1 + q2], axis=1)


def settling_time(traj: Trajectory, angle_tol: float = np.deg2rad(2.0),
                  pos_tol: float = 0.05) -> float:
    """
    TODO(you): first time after which the system STAYS within tolerance until the end.
    Return np.inf if it never settles.
    """
    raise NotImplementedError


def rms_position_error(traj: Trajectory) -> float:
    """TODO(you): RMS of the drone's (y, z) offset from the target."""
    raise NotImplementedError


def max_and_rms_swing(traj: Trajectory) -> tuple[float, float]:
    """TODO(you): max and RMS absolute link angle (which link? both? report both)."""
    raise NotImplementedError


def control_effort(traj: Trajectory) -> float:
    """TODO(you): e.g. mean squared action, or integral of |thrust - hover|."""
    raise NotImplementedError
