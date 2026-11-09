
"""Loads the MJCF file and provides access to the underlying MuJoCo objects."""

import mujoco
import numpy as np


class DronePendulumModel:
    """Loads and provides access to the drone-pendulum simulation model."""

    def __init__(self, model_path: str) -> None:
        """
        Initialize the MuJoCo model and resolve component IDs.

        Parameters
        ----------
        model_path : str
            Path to the MJCF model file.
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.drone_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "drone"
        )

        self.drone_free_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "drone_free"
        )
        self.joint1_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint1"
        )
        self.joint2_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint2"
        )

        # Resolve generalized position addresses
        self.joint1_qpos_adr = self.model.jnt_qposadr[self.joint1_id]
        self.joint2_qpos_adr = self.model.jnt_qposadr[self.joint2_id]

        # Resolve generalized velocity addresses
        self.drone_dof_adr = self.model.jnt_dofadr[self.drone_free_id]
        self.joint1_dof_adr = self.model.jnt_dofadr[self.joint1_id]
        self.joint2_dof_adr = self.model.jnt_dofadr[self.joint2_id]
        
        self.total_mass = float(self.model.body_subtreemass[self.drone_id]) # (kg)
        
        mujoco.mj_forward(self.model, self.data)

    def get_drone_position(self) -> np.ndarray:
        """Return the drone's world-frame position."""
        return self.data.xpos[self.drone_id].copy()

    def get_drone_orientation(self) -> np.ndarray:
        """Return the drone's world-frame orientation as a quaternion."""
        return self.data.xquat[self.drone_id].copy()

    def get_drone_angular_velocity(self, frame: str = "body") -> np.ndarray:
        """
        Return the drone's angular velocity in the requested frame.

        Parameters
        ----------
        frame : str, default="body"
            Coordinate frame: "body" or "world".

        Returns
        -------
        np.ndarray, shape (3,)
            Angular velocity in rad/s.
        """
        angular_velocity_body = self.data.qvel[
            self.drone_dof_adr + 3:self.drone_dof_adr + 6
        ]

        if frame == "body":
            return angular_velocity_body.copy()

        if frame == "world":
            rotation = self.data.xmat[self.drone_id].reshape(3, 3)
            return rotation @ angular_velocity_body

        raise ValueError(f"Unsupported coordinate frame: {frame}")

    def get_drone_linear_velocity(self) -> np.ndarray:
        """Return the drone's linear velocity in the global frame."""
        return self.data.qvel[
            self.drone_dof_adr:self.drone_dof_adr + 3
        ].copy()

    def get_joint_positions(self) -> np.ndarray:
        """Return the two pendulum joint angles in radians."""
        return np.array([
            self.data.qpos[self.joint1_qpos_adr],
            self.data.qpos[self.joint2_qpos_adr],
        ])

    def get_joint_velocities(self) -> np.ndarray:
        """Return the two pendulum joint angular velocities in rad/s."""
        return np.array([
            self.data.qvel[self.joint1_dof_adr],
            self.data.qvel[self.joint2_dof_adr],
        ])
