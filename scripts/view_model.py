"""Script for visualizing the MuJoCo model"""
import time

import mujoco
import mujoco.viewer

MODEL_PATH = "src/drone_pendulum/simulation/assets/drone_pendulum.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Find the drone and pendulum joints
drone_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "drone"
)
joint1_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "joint1"
)
joint2_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "joint2"
)

# Initial joint angles (radians)
data.qpos[model.jnt_qposadr[joint1_id]] = 1
data.qpos[model.jnt_qposadr[joint2_id]] = -1.7

# Initial Joint Velocities
data.qvel[model.jnt_dofadr[joint1_id]] = 0
data.qvel[model.jnt_dofadr[joint2_id]] = 0


# Initial Drone Velocity
data.qvel[0] = 0.0  # X velocity (m/s)
data.qvel[1] = 0.0  # Y velocity (m/s)

mujoco.mj_forward(model, data)

# Total system mass: 1.0 + 0.15 + 0.10 kg
hover_thrust = 1.25 * 9.81


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.xfrc_applied[drone_id, 2] = hover_thrust

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)

