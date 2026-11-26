"""Run the drone and double pendulum with one randomly sampled command."""

import argparse
from contextlib import contextmanager, nullcontext
import math
import threading
import time

import mujoco
import numpy as np

from drone_pendulum.envs.registration import STARLING_MODEL_PATH
from drone_pendulum.simulation.ideal_rate_simulator import IdealRateSimulator


@contextmanager
def open_viewer(sim: IdealRateSimulator):
    from mujoco.viewer import launch_passive

    existing_threads = set(threading.enumerate())
    viewer = launch_passive(sim.model, sim.data)
    viewer_threads = [
        thread for thread in threading.enumerate()
        if thread.daemon and thread not in existing_threads
    ]
    try:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.drone_model.drone_id
            viewer.cam.distance = 2.5
        yield viewer
    finally:
        viewer.close()
        # The passive viewer closes asynchronously. Let its render thread
        # destroy the GL context before GLFW's interpreter-exit cleanup.
        for thread in viewer_threads:
            thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation seconds (default: 60)")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True,
                        help="Open the MuJoCo viewer (default: on)")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a positive finite number")

    # The demo prescribes angular motion, which can fight collision impulses.
    sim = IdealRateSimulator(str(STARLING_MODEL_PATH), drone_body_name="starling2max")
    sim.reset()
    free_joint_start = sim.model.jnt_qposadr[sim.drone_model.drone_free_id]
    sim.data.qpos[free_joint_start + 2] = 1.0
    mujoco.mj_forward(sim.model, sim.data)
    rng = np.random.default_rng()
    hover_thrust = sim.drone_model.total_mass * abs(sim.model.opt.gravity[2])
    thrust = hover_thrust * rng.uniform(0.95, 1.15)
    body_rates = rng.uniform(-0.35, 0.35, size=3)
    action = np.array([thrust, *body_rates])
    print(f"Fixed command [T (N), p, q, r (rad/s)] = {action}", flush=True)

    timestep = sim.model.opt.timestep
    with open_viewer(sim) if args.render else nullcontext() as viewer:
        for _ in range(math.ceil(args.duration / timestep)):
            if viewer is not None and not viewer.is_running():
                break
            tick_start = time.monotonic()
            with viewer.lock() if viewer is not None else nullcontext():
                sim.step(action)
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, timestep - (time.monotonic() - tick_start)))

    print(f"Final state at {sim.data.time:.3f} s")
    for name, value in sim.get_state().items():
        print(f"  {name} = {value}")


if __name__ == "__main__":
    main()
