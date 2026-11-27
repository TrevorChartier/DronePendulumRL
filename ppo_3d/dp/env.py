"""Gymnasium env: 3D Starling-class quadrotor carrying a passive double pendulum.

Built so the trained policy can be deployed on a ModalAI Starling 2 Max through
PX4 offboard mode with an attitude-rate setpoint:

    action (policy output, [-1, 1]^4) -> thrust_norm in [0, 1], body rates (rad/s)

`DronePendulumEnv.action_to_setpoint` and `DronePendulumEnv.build_obs` are the
conversions used both in sim and on hardware, so the action scaling and
observation layout never have to be re-derived.

Sim pipeline (what a real flight stack does between the setpoint and the props):
    setpoint -> latency queue -> body-rate P loop (bandwidth-set) -> mixer
             -> per-rotor thrust -> RPM clip -> first-order rotor speed (asym up/down)
             -> F = kT w^2 -> MuJoCo
Drone attitude is fully physical, so the pendulum's reaction torques act on it.

Frames: world is MuJoCo ENU (z up); body is the CAD frame, FLU (x along the long
motor axis). PX4 uses NED/FRD: on hardware send p_frd = p, q_frd = -q, r_frd = -r,
and convert VIO/mocap states to ENU/FLU before building the observation.

Domain randomization covers what differs between sim and the real rig: masses
(m2 especially), per-rotor thrust scale (battery sag), motor time constants,
rate-loop gains, joint friction, control latency, and sensor noise.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import os

MODEL_PATH = Path(os.environ.get("DP_MODEL", Path(__file__).resolve().parent.parent / "assets" / "drone_pendulum.xml"))

POLICY_DT = 0.02  # 50 Hz policy / offboard setpoint rate
MAX_RATE = np.array([6.0, 6.0, 2.0])  # rad/s at |action| = 1 for roll, pitch, yaw rate

# Starling 2 Max (D0012) propulsion, from the team repo's prop_model.py.
KT = 3.964738e-6  # N / (rad/s)^2
OMEGA_MIN = 2000.0 * 2 * np.pi / 60
OMEGA_MAX = 12000.0 * 2 * np.pi / 60
F_MAX = KT * OMEGA_MAX ** 2  # 6.26 N per rotor
TAU_UP, TAU_DOWN = 0.055, 0.085

# Body-rate loop: torque = I_drone * RATE_BW * (w_des - w) + w x I w. Bandwidth (1/s)
# stands in for PX4's tuned rate PID; randomized +/-30% per episode.
RATE_BW = np.array([20.0, 20.0, 10.0])
# Integral term (1/s^2), like PX4's MC_*RATE_I: rejects constant torques such as
# per-rotor thrust imbalance. Set rate_integral=False for the old P-only loop.
RATE_KI = np.array([10.0, 10.0, 4.0])
RATE_I_LIMIT = 0.3  # rad of accumulated rate error

# Sensor noise std: pos (m), vel (m/s), attitude (rad), gyro (rad/s),
# joint angle (rad), joint rate (rad/s).
NOISE = dict(pos=0.005, vel=0.02, att=0.008, gyro=0.02, q=0.008, qd=0.08)


class DronePendulumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / POLICY_DT)}

    def __init__(
        self,
        render_mode: str | None = None,
        episode_seconds: float = 10.0,
        randomize: bool = True,
        init_angle: float = 1.2,
        init_rate: float = 3.0,
        camera: str = "iso",
        m2_range: tuple[float, float] = (0.049, 0.150),
        max_latency_steps: int = 2,
        leash_radius: float = 0.0,
        boundary: float = 2.5,
        w_pos: float = 1.5,
        w_swing: float = 1.0,
        rate_integral: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.randomize = randomize
        self.init_angle = init_angle
        self.init_rate = init_rate
        self.camera = camera
        self.m2_range = m2_range
        self.max_latency_steps = max_latency_steps
        # Position anchoring: free inside leash_radius, linear penalty w_pos beyond it,
        # episode ends past `boundary` (think: edge of the mocap volume).
        self.leash_radius = leash_radius
        self.boundary = boundary
        self.w_pos = w_pos
        self.w_swing = w_swing
        self.rate_integral = rate_integral
        self._renderer = None

        m = self.model
        self.n_sub = round(POLICY_DT / m.opt.timestep)
        self.max_steps = round(episode_seconds / POLICY_DT)
        self.drone_id = m.body("drone").id
        self.link_ids = [m.body("link1").id, m.body("link2").id]
        self.m2_id = m.body("m2").id
        self.site_ids = [m.site(n).id for n in ("pivot1", "pivot2", "tip")]
        self.target = m.site("target").pos.copy()
        self.jq = m.jnt_qposadr[m.joint("joint1").id]
        self.jv = m.jnt_dofadr[m.joint("joint1").id]

        self.nom_mass = m.body_mass.copy()
        self.nom_inertia = m.body_inertia.copy()

        # Mixer from rotor geometry: [T, tx, ty, tz] = A @ f
        pos = np.array([m.site_pos[m.actuator_trnid[i, 0]] for i in range(m.nu)])
        A = np.vstack([np.ones(m.nu), pos[:, 1], -pos[:, 0], m.actuator_gear[:, 5]])
        self.mix_inv = np.linalg.inv(A)
        self.drone_I = self.nom_inertia[self.drone_id].copy()

        # Nominal take-off mass with m2 at the middle of its range (XML value).
        self.nominal_mass = float(self.nom_mass[self.drone_id:].sum())
        self.hover_thrust_norm = self.nominal_mass * 9.81 / (4 * F_MAX)

        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (32,), np.float32)

    # ------------------------------------------------------------ deployment API
    def action_to_setpoint(self, action):
        """Policy action -> (normalized collective thrust in [0, 1], body rates FLU rad/s)."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        thrust_norm = float(np.clip(self.hover_thrust_norm * (1.0 + a[0]), 0.0, 1.0))
        return thrust_norm, a[1:] * MAX_RATE

    @staticmethod
    def build_obs(pos_err, vel, R, gyro, q, qd, prev_actions):
        """Observation layout shared by sim and hardware (world ENU, body FLU).

        pos_err = position - target (3), vel world (3), R body->world row-major (9),
        gyro body (3), sin/cos of joint angles (4), joint rates (2), last two actions (8).
        """
        return np.concatenate([
            pos_err, vel, np.asarray(R).reshape(-1), gyro,
            np.sin(q), np.cos(q), qd, np.asarray(prev_actions).reshape(-1),
        ]).astype(np.float32)

    # ----------------------------------------------------------------- helpers
    def _episode_params(self):
        r, rnd, m = self.np_random, self.randomize, self.model

        def u(lo, hi, n=None):
            if rnd:
                return r.uniform(lo, hi, n)
            return (lo + hi) / 2 if n is None else np.full(n, (lo + hi) / 2)

        m.body_mass[:] = self.nom_mass
        m.body_inertia[:] = self.nom_inertia
        if rnd:
            for b in [self.drone_id, *self.link_ids]:
                s = r.uniform(0.9, 1.1)
                m.body_mass[b] *= s
                m.body_inertia[b] *= s
        self.thrust_scale = u(0.9, 1.08, 4)  # battery sag / prop variation, per rotor
        self.tau_up, self.tau_down = u(0.04, 0.07), u(0.07, 0.10)
        self.k_omega = self.drone_I * RATE_BW * u(0.7, 1.3, 3)
        self.latency = int(r.integers(0, self.max_latency_steps + 1)) if rnd else 1
        m.dof_damping[self.jv:self.jv + 2] = u(0.0, 3e-4, 2)
        self.noise_scale = 1.0 if rnd else 0.0

    def _set_m2(self, m2):
        self.m2 = float(m2)
        self.model.body_mass[self.m2_id] = self.m2
        self.model.body_inertia[self.m2_id] = 0.4 * self.m2 * 0.012 ** 2

    def _rate_control(self, thrust_norm, w_des):
        w = self.data.qvel[3:6]  # free-joint angular velocity is body-frame
        err = w_des - w
        tau = self.k_omega * err + np.cross(w, self.drone_I * w)
        if self.rate_integral:
            dt = self.model.opt.timestep
            self.rate_int = np.clip(self.rate_int + err * dt, -RATE_I_LIMIT, RATE_I_LIMIT)
            tau += self.drone_I * RATE_KI * self.rate_int
        f_cmd = self.mix_inv @ np.concatenate([[thrust_norm * 4 * F_MAX], tau])
        omega_cmd = np.sqrt(np.clip(f_cmd, KT * OMEGA_MIN ** 2, F_MAX) / KT)
        dt = self.model.opt.timestep
        tau_m = np.where(omega_cmd > self.omega, self.tau_up, self.tau_down)
        self.omega += (omega_cmd - self.omega) * (1.0 - np.exp(-dt / tau_m))
        return self.thrust_scale * KT * self.omega ** 2

    def _link_dirs(self):
        p = self.data.site_xpos[self.site_ids]
        d = np.diff(p, axis=0)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def _rotvec_quat(v):
        q = np.zeros(4)
        ang = np.linalg.norm(v)
        mujoco.mju_axisAngle2Quat(q, v / ang if ang > 0 else np.array([1.0, 0, 0]), ang)
        return q

    def _obs(self):
        d, r, k = self.data, self.np_random, self.noise_scale

        def n(key, size):
            return r.normal(0.0, NOISE[key] * k, size)

        R = d.xmat[self.drone_id].reshape(3, 3)
        if k:
            dR = np.zeros(9)
            mujoco.mju_quat2Mat(dR, self._rotvec_quat(n("att", 3)))
            R = R @ dR.reshape(3, 3)
        return self.build_obs(
            d.qpos[:3] - self.target + n("pos", 3),
            d.qvel[:3] + n("vel", 3),
            R,
            d.qvel[3:6] + n("gyro", 3),
            d.qpos[self.jq:self.jq + 2] + n("q", 2),
            d.qvel[self.jv:self.jv + 2] + n("qd", 2),
            list(self.action_hist),
        )

    # --------------------------------------------------------------------- API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r = self.np_random
        options = options or {}
        self._episode_params()
        self._set_m2(options.get("m2", r.uniform(*self.m2_range)))
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[:3] = self.target + r.uniform(-0.3, 0.3, 3)
        axis = r.normal(size=3)
        q_tilt = self._rotvec_quat(axis / np.linalg.norm(axis) * r.uniform(0, np.deg2rad(15)))
        q_yaw = self._rotvec_quat(np.array([0, 0, r.uniform(-np.pi, np.pi)]))
        mujoco.mju_mulQuat(d.qpos[3:7], q_yaw, q_tilt)
        d.qvel[:3] = r.uniform(-0.5, 0.5, 3)
        d.qvel[3:6] = r.uniform(-0.5, 0.5, 3)
        d.qpos[self.jq:self.jq + 2] = r.uniform(-self.init_angle, self.init_angle, 2)
        d.qvel[self.jv:self.jv + 2] = r.uniform(-self.init_rate, self.init_rate, 2)
        mujoco.mj_forward(self.model, d)

        hover = self.model.body_mass[self.drone_id:].sum() * 9.81 / 4
        self.omega = np.full(4, np.sqrt(hover / KT))
        self.rate_int = np.zeros(3)
        self.action_hist = deque([np.zeros(4), np.zeros(4)], maxlen=2)
        self.pending = deque([np.zeros(4)] * self.latency)
        self.steps = 0
        return self._obs(), {"m2": self.m2}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.pending.append(a)
        applied = self.pending.popleft()  # latency: setpoint reaches the FC late
        thrust_norm, w_des = self.action_to_setpoint(applied)
        for _ in range(self.n_sub):
            self.data.ctrl[:] = self._rate_control(thrust_norm, w_des)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        d = self.data
        pos_err = np.linalg.norm(d.qpos[:3] - self.target)
        vel = np.linalg.norm(d.qvel[:3])
        dirs = self._link_dirs()
        swing = 1.0 + dirs[:, 2]  # 0 hanging straight down, 2 inverted
        swing_rate = np.sum(d.cvel[self.link_ids, :2] ** 2)  # world horizontal-axis spin
        body_rate = np.sum(d.qvel[3:6] ** 2)
        up_z = d.xmat[self.drone_id][8]
        da = np.sum((a - self.action_hist[-1]) ** 2)

        reward = (
            2.0
            - self.w_pos * max(0.0, pos_err - self.leash_radius)
            - 0.2 * vel
            - self.w_swing * swing.sum()
            - 0.03 * swing_rate
            - 0.02 * body_rate
            - 0.5 * (1.0 - up_z)
            - 0.1 * da
        )
        self.action_hist.append(a)

        terminated = bool(pos_err > self.boundary or up_z < 0.0 or not np.all(np.isfinite(d.qpos)))
        if terminated:
            reward -= 50.0
        truncated = self.steps >= self.max_steps
        info = {
            "pos_err": pos_err,
            "m2": self.m2,
            "thrust_norm": thrust_norm,
            "swing_deg": np.degrees(np.arccos(np.clip(-dirs[:, 2], -1, 1))),
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
