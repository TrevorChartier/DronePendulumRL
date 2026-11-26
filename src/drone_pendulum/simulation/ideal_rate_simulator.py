"""MuJoCo simulation with instantaneous body-rate tracking."""

import mujoco
import numpy as np

from drone_pendulum.simulation.simulator import DronePendulumSimulator


class IdealRateSimulator(DronePendulumSimulator):
    """Accept collective thrust in newtons and body rates in rad/s."""

    def step(self, action: np.ndarray, n_substeps: int = 1) -> None:
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected four finite values: [thrust, p, q, r].")
        if action[0] < 0:
            raise ValueError("Collective thrust must be nonnegative.")
        if n_substeps < 1:
            raise ValueError("n_substeps must be at least 1.")

        drone_id = self.drone_model.drone_id
        free_joint = self.drone_model.drone_free_id
        pos_start = self.model.jnt_qposadr[free_joint]
        quat = self.data.qpos[pos_start + 3:pos_start + 7]
        vel_start = self.drone_model.drone_dof_adr
        body_rates = self.data.qvel[vel_start + 3:vel_start + 6]
        thrust, rates = action[0], action[1:]
        timestep = self.model.opt.timestep

        for _ in range(n_substeps):
            body_rates[:] = rates
            next_quat = quat.copy()
            mujoco.mju_quatIntegrate(next_quat, rates, timestep)

            # Use the current attitude for thrust during this physics tick.
            wrench = self.data.xfrc_applied[drone_id]
            wrench[:3] = thrust * self.data.xmat[drone_id].reshape(3, 3)[:, 2]
            wrench[3:] = 0
            mujoco.mj_step(self.model, self.data)

            # Prescribe drone rotation while MuJoCo evolves translation and
            # the passive pendulum joints.
            quat[:] = next_quat
            body_rates[:] = rates
            mujoco.mj_forward(self.model, self.data)
