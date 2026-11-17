"""Drone-only port of the drone-soccer native MuJoCo dynamics backend."""

from abc import ABC, abstractmethod
from pathlib import Path

import mujoco
import numpy as np

from drone_pendulum.starling_simulation.prop_model import (
    thrust_commands_to_rotor_thrusts,
)
from drone_pendulum.starling_simulation.state import DroneState, PendulumWorldState


def _zero_out_drone_fluid(model, drone_body_id):
    """Match drone_soccer's zero-fluid setup for the drone body."""
    drone_geoms = np.flatnonzero(model.geom_bodyid == drone_body_id)
    model.geom_fluid[drone_geoms, :] = 0.0
    model.geom_fluid[drone_geoms, 0] = 1.0


class Dynamics(ABC):
    @property
    @abstractmethod
    def sim_dt(self): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def step(self, state, motor_thrusts): ...


class StarlingPendulumDynamics(Dynamics):
    """Native MuJoCo dynamics with soccer's drone behavior plus pendulum state."""

    def __init__(self, scene_path: str | Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.drone_body_id = int(self.model.body("starling2max").id)
        drone_joint_id = int(self.model.body("starling2max").jntadr[0])
        self.drone_qpos_adr = int(self.model.jnt_qposadr[drone_joint_id])
        self.drone_qvel_adr = int(self.model.jnt_dofadr[drone_joint_id])
        self.joint_ids = [int(self.model.joint(name).id) for name in ("joint1", "joint2")]
        self.joint_qpos_adrs = [int(self.model.jnt_qposadr[i]) for i in self.joint_ids]
        self.joint_dof_adrs = [int(self.model.jnt_dofadr[i]) for i in self.joint_ids]
        _zero_out_drone_fluid(self.model, self.drone_body_id)
        self.data = mujoco.MjData(self.model)
        self._last_state = None
        self._hover_key_id = int(self.model.key("hover").id)
        self._renderer = None
        self._camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._camera)
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.distance = 2.5
        self._camera.azimuth = 135.0
        self._camera.elevation = -20.0

    @property
    def sim_dt(self):
        return float(self.model.opt.timestep)

    def reset(self):
        """Reset using the soccer backend's keyframe plus its hover pose."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._hover_key_id)
        self.data.qpos[self.drone_qpos_adr:self.drone_qpos_adr + 3] = [0.0, 0.0, 1.0]
        self.data.qpos[self.drone_qpos_adr + 3:self.drone_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self.model.opt.wind[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_state = self._extract_state()
        return self._last_state

    def set_state(self, state):
        drone_qpos = self.drone_qpos_adr
        self.data.qpos[drone_qpos:drone_qpos + 3] = state.drone.position
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(state.drone.rotation).flatten())
        self.data.qpos[drone_qpos + 3:drone_qpos + 7] = quat
        drone_qvel = self.drone_qvel_adr
        self.data.qvel[drone_qvel:drone_qvel + 3] = state.drone.velocity
        self.data.qvel[drone_qvel + 3:drone_qvel + 6] = state.drone.angular_velocity
        for qpos_adr, dof_adr, position, velocity in zip(
            self.joint_qpos_adrs,
            self.joint_dof_adrs,
            state.joint_positions,
            state.joint_velocities,
        ):
            self.data.qpos[qpos_adr] = position
            self.data.qvel[dof_adr] = velocity
        self.data.time = float(state.t)
        mujoco.mj_forward(self.model, self.data)
        self._last_state = state

    def step(self, state, motor_thrusts):
        if state is not self._last_state:
            self.set_state(state)
        thrusts = np.asarray(motor_thrusts, dtype=float)
        if thrusts.shape != (4,):
            raise ValueError(f"motor_thrusts must have shape (4,), got {thrusts.shape}")
        self.data.ctrl[:4] = thrust_commands_to_rotor_thrusts(thrusts)
        mujoco.mj_step(self.model, self.data)
        self._last_state = self._extract_state()
        return self._last_state

    def render(self, width=640, height=480):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        drone_position = self.data.xpos[self.drone_body_id]
        self._camera.lookat[:] = drone_position + np.array([0.0, 0.0, -0.35])
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _extract_state(self):
        rotation_flat = np.zeros(9)
        mujoco.mju_quat2Mat(rotation_flat, self.data.xquat[self.drone_body_id])
        drone_qvel = self.drone_qvel_adr
        drone = DroneState(
            position=self.data.xpos[self.drone_body_id].copy(),
            velocity=self.data.qvel[drone_qvel:drone_qvel + 3].copy(),
            rotation=rotation_flat.reshape(3, 3),
            angular_velocity=self.data.qvel[drone_qvel + 3:drone_qvel + 6].copy(),
        )
        return PendulumWorldState(
            drone=drone,
            joint_positions=np.array([self.data.qpos[i] for i in self.joint_qpos_adrs]),
            joint_velocities=np.array([self.data.qvel[i] for i in self.joint_dof_adrs]),
            t=float(self.data.time),
        )
