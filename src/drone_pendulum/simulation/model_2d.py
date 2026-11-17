"""MuJoCo indexing helpers for the planar (2D) drone + double pendulum.

The planar model has exactly 5 scalar joints, so the generalized coordinates are simply

    qpos = [y, z, roll, q1, q2]
    qvel = [y_dot, z_dot, roll_dot, q1_dot, q2_dot]

where y/z are the drone's displacement from its spawn point (the slide joints start at 0),
roll is the drone attitude, and q1/q2 are the RELATIVE pendulum joint angles
(see the comment at the top of assets/drone_pendulum_2d.xml).

Throughout the planar code, the "state vector" means the 10-vector
    x = [qpos, qvel] = [y, z, roll, q1, q2, y_dot, z_dot, roll_dot, q1_dot, q2_dot]
This is the vector you will linearize around for LQR.
"""

import mujoco
import numpy as np

# Names of the generalized coordinates, in state-vector order. Handy for plotting/logging.
STATE_NAMES = (
    "y", "z", "roll", "q1", "q2",
    "y_dot", "z_dot", "roll_dot", "q1_dot", "q2_dot",
)
NQ = 5
STATE_DIM = 2 * NQ


class PlanarDronePendulumModel:
    """Loads the planar MJCF and provides named access to its state."""

    JOINT_NAMES = ("drone_y", "drone_z", "drone_roll", "joint1", "joint2")

    def __init__(self, model_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        if self.model.nq != NQ or self.model.nv != NQ:
            raise ValueError(
                f"Expected a planar model with nq = nv = {NQ}, "
                f"got nq={self.model.nq}, nv={self.model.nv}"
            )

        self.drone_id = self._body_id("drone")
        self.link1_id = self._body_id("link1")
        self.link2_id = self._body_id("link2")

        # Sanity check: joints are declared in the order we assume above.
        for i, name in enumerate(self.JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if self.model.jnt_qposadr[jid] != i or self.model.jnt_dofadr[jid] != i:
                raise ValueError(f"Joint '{name}' is not at index {i} in qpos/qvel")

        # Spawn position of the drone in world (y, z); slide joints are offsets from this.
        self.spawn_yz = self.model.body_pos[self.drone_id][1:3].copy()

        self.total_mass = float(self.model.body_subtreemass[self.drone_id])  # kg
        self.gravity = float(-self.model.opt.gravity[2])  # m/s^2 (positive)

        # Pendulum geometry, read from the model so it stays correct if you edit the XML.
        self.link1_length = float(-self.model.body_pos[self.link2_id][2])
        self.link2_length = float(-self.model.geom("link2_mass").pos[2])

        mujoco.mj_forward(self.model, self.data)

    def _body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    # ------------------------------------------------------------------ state vector

    def get_state_vector(self) -> np.ndarray:
        """Return x = [y, z, roll, q1, q2, y_dot, z_dot, roll_dot, q1_dot, q2_dot]."""
        return np.concatenate([self.data.qpos, self.data.qvel]).copy()

    def set_state_vector(self, x: np.ndarray) -> None:
        """Overwrite the full simulator state with x (see get_state_vector)."""
        x = np.asarray(x, dtype=float)
        if x.shape != (STATE_DIM,):
            raise ValueError(f"State must have shape ({STATE_DIM},), got {x.shape}")
        self.data.qpos[:] = x[:NQ]
        self.data.qvel[:] = x[NQ:]
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ named accessors

    def get_drone_position(self) -> np.ndarray:
        """Drone (y, z) in the world frame, in metres."""
        return self.data.xpos[self.drone_id][1:3].copy()

    def get_drone_velocity(self) -> np.ndarray:
        """Drone (y_dot, z_dot) in the world frame, in m/s."""
        return self.data.qvel[0:2].copy()

    def get_drone_roll(self) -> float:
        return float(self.data.qpos[2])

    def get_drone_roll_rate(self) -> float:
        return float(self.data.qvel[2])

    def get_joint_positions(self) -> np.ndarray:
        """Relative pendulum joint angles [q1, q2] in radians."""
        return self.data.qpos[3:5].copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Relative pendulum joint rates [q1_dot, q2_dot] in rad/s."""
        return self.data.qvel[3:5].copy()

    def get_absolute_link_angles(self) -> np.ndarray:
        """World-frame link angles [phi1, phi2] from vertical (0 = hanging down)."""
        roll, q1, q2 = self.data.qpos[2:5]
        return np.array([roll + q1, roll + q1 + q2])

    def get_absolute_link_rates(self) -> np.ndarray:
        """Time derivatives of get_absolute_link_angles()."""
        roll_dot, q1_dot, q2_dot = self.data.qvel[2:5]
        return np.array([roll_dot + q1_dot, roll_dot + q1_dot + q2_dot])

    def get_payload_positions(self) -> np.ndarray:
        """World (y, z) of the two point masses, shape (2, 2): [[y1, z1], [y2, z2]]."""
        m1 = self.data.geom("link1_mass").xpos[1:3]
        m2 = self.data.geom("link2_mass").xpos[1:3]
        return np.array([m1, m2])
