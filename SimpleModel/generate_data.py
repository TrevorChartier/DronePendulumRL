"""Generate independent Starling transitions and held-out action rollouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

import mujoco
import numpy as np

from drone_pendulum.envs.registration import STARLING_MODEL_PATH
from drone_pendulum.simulation.ideal_rate_simulator import IdealRateSimulator


CONTROL_DT = 0.1
STATE_NAMES = (
    "roll", "pitch", "cos_yaw", "sin_yaw", "vx", "vy", "vz",
    "theta1", "theta2", "omega1", "omega2",
)


def reset_pose(sim, rng):
    model, data = sim.model, sim.data
    mujoco.mj_resetData(model, data)
    drone = sim.drone_model
    qpos_start = model.jnt_qposadr[drone.drone_free_id]
    dof_start = drone.drone_dof_adr
    data.qpos[qpos_start:qpos_start + 3] = [0, 0, 5]
    roll, pitch = np.deg2rad(rng.uniform(-30, 30, size=2))
    yaw = rng.uniform(0, 2 * np.pi)
    r, p, y = roll / 2, pitch / 2, yaw / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    data.qpos[qpos_start + 3:qpos_start + 7] = [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
    data.qvel[dof_start:dof_start + 3] = rng.uniform(-1, 1, size=3)
    data.qpos[[drone.joint1_qpos_adr, drone.joint2_qpos_adr]] = np.deg2rad(
        rng.uniform([-50, -120], [50, 120])
    )
    data.qvel[[drone.joint1_dof_adr, drone.joint2_dof_adr]] = np.deg2rad(
        rng.uniform(-60, 60, size=2)
    )
    mujoco.mj_forward(model, data)


def observation(sim):
    drone = sim.drone_model
    rot = sim.data.xmat[drone.drone_id].reshape(3, 3)
    roll = math.atan2(rot[2, 1], rot[2, 2])
    pitch = math.atan2(-rot[2, 0], math.hypot(rot[0, 0], rot[1, 0]))
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return np.array([
        roll, pitch, math.cos(yaw), math.sin(yaw),
        *drone.get_drone_linear_velocity(),
        *drone.get_joint_positions(), *drone.get_joint_velocities(),
    ], dtype=np.float32)


def generate_chunk(job):
    count, length, seed = job
    rng = np.random.default_rng(seed)
    scene = STARLING_MODEL_PATH.with_name("starling2max_pendulum_floor.xml")
    sim = IdealRateSimulator(str(scene), drone_body_name="starling2max")
    hover_thrust = sim.drone_model.total_mass * -sim.model.opt.gravity[2]
    substeps = round(CONTROL_DT / sim.model.opt.timestep)
    if not math.isclose(substeps * sim.model.opt.timestep, CONTROL_DT):
        raise ValueError("Control interval must be a multiple of the physics timestep.")

    states = np.empty((count, length + 1, len(STATE_NAMES)), dtype=np.float32)
    actions = np.empty((count, length, 4), dtype=np.float32)
    for index in range(count):
        reset_pose(sim, rng)
        states[index, 0] = observation(sim)
        for step in range(length):
            action = [
                hover_thrust * rng.uniform(0.6, 1.4),
                *np.deg2rad(rng.uniform(-20, 20, size=3)),
            ]
            actions[index, step] = action
            sim.step(actions[index, step], n_substeps=substeps)
            if sim.data.xpos[sim.drone_model.drone_id, 2] < 0.7:
                raise RuntimeError("Rollout entered the floor-contact region.")
            states[index, step + 1] = observation(sim)
    if not np.isfinite(states).all():
        raise RuntimeError("MuJoCo produced a non-finite observation.")
    return states, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("SimpleModel/runs/data.npz"))
    parser.add_argument("--train", type=int, default=250_000)
    parser.add_argument("--validation", type=int, default=15_000)
    parser.add_argument("--test", type=int, default=15_000)
    parser.add_argument("--rollouts", type=int, default=2048, help="Per held-out split")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test, args.rollouts, args.workers) < 1:
        parser.error("Dataset sizes and worker count must be positive.")

    splits = (
        ("train", args.train, 1),
        ("validation", args.validation, 1),
        ("test", args.test, 1),
        ("validation_rollout", args.rollouts, 10),
        ("test_rollout", args.rollouts, 10),
    )
    rng = np.random.default_rng(args.seed)
    dataset = {"control_dt": CONTROL_DT, "state_names": STATE_NAMES}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, count, length in splits:
            sizes = [count // args.workers] * args.workers
            for index in range(count % args.workers):
                sizes[index] += 1
            jobs = [(size, length, int(rng.integers(2**63))) for size in sizes if size]
            results = list(pool.map(generate_chunk, jobs))
            dataset[f"{name}_states"] = np.concatenate([pair[0] for pair in results])
            dataset[f"{name}_actions"] = np.concatenate([pair[1] for pair in results])
            print(f"{name}: {count:,} samples × {length} steps", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **dataset)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
