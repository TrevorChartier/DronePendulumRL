import numpy as np


def compute_abs_waypoints(drone_positions, actions):
    drone_positions = np.asarray(drone_positions, dtype=float)
    actions = np.asarray(actions, dtype=float)
    return np.stack(
        [
            drone_positions[:, 0] + actions[:, 0],
            drone_positions[:, 1] + actions[:, 1],
            actions[:, 2],
        ],
        axis=-1,
    )
