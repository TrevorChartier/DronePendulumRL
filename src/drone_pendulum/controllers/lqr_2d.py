"""LQR baseline for the planar drone + double pendulum (primary baseline in the proposal).

Plan:
  1. Linearize the closed-loop-with-inner-rate-controller dynamics about hover:
         x_{k+1} - x_eq  ~=  A (x_k - x_eq) + B (u_k - u_eq)
     with x the 10-vector state (see simulation/model_2d.py) and u the 2-vector
     normalized action, at the ENV's control rate (env.control_dt), not the physics rate.
  2. Pick Q (10x10) and R (2x2).
  3. Solve the discrete-time algebraic Riccati equation for K.
  4. Control law:  u = u_eq - K (x - x_eq), clipped to [-1, 1].
"""

import numpy as np

from drone_pendulum.controllers.base import Controller


def linearize(env, x_eq: np.ndarray, u_eq: np.ndarray, eps: float = 1e-5):
    """
    Numerically linearize one env control step around (x_eq, u_eq).

    Returns
    -------
    A : (10, 10) ndarray
    B : (10, 2) ndarray

    Approach: central finite differences through the simulator.
      - f(x, u): env.sim.reset(x); env.sim.step(u, n_substeps=env.frame_skip);
                 return env.sim.drone_model.get_state_vector()
      - A[:, i] = (f(x_eq + eps*e_i, u_eq) - f(x_eq - eps*e_i, u_eq)) / (2*eps)
      - B[:, j] likewise, perturbing u.
      - Sanity check: f(x_eq, u_eq) should equal x_eq (it's an equilibrium!). If it
        doesn't, your u_eq is wrong.

    Note: mujoco.mjd_transitionFD will NOT work out of the box here, because our
    actions are applied through data.xfrc_applied rather than MuJoCo actuators/ctrl.
    (Converting the model to use <actuator> elements is a valid alternative.)

    Derive A, B analytically from the Lagrangian and compare against this.
    """
    def f(x, u):
        env.sim.reset(x)
        env.sim.step(u, n_substeps=env.frame_skip)
        return env.sim.drone_model.get_state_vector()
    A = np.zeros((10, 10))
    for i in range(10):
        e_i = np.zeros(10)
        e_i[i] = 1.0
        A[:, i] = (f(x_eq + eps * e_i, u_eq) - f(x_eq - eps * e_i, u_eq)) / (2 * eps)
    B = np.zeros((10, 2))
    for j in range(2):
        e_j = np.zeros(2)
        e_j[j] = 1.0
        B[:, j] = (f(x_eq, u_eq + eps * e_j) - f(x_eq, u_eq - eps * e_j)) / (2 * eps)
    assert np.allclose(f(x_eq, u_eq), x_eq, atol=1e-5), "f(x_eq, u_eq) != x_eq; check your u_eq"
    env.sim.reset(x_eq)  # reset to equilibrium for safety
    return A, B


def design_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Return the (2, 10) discrete-time LQR gain K.
    scipy.linalg.solve_discrete_are, then K = (R + B^T P B)^-1 B^T P A.
    Check: eigenvalues of (A - B K) should all lie strictly inside the unit circle.
    And is (A, B) controllable? (rank of the controllability matrix)
    """
    from scipy.linalg import solve_discrete_are

    # check controllability
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])
    if np.linalg.matrix_rank(C) < n:
        raise ValueError("(A, B) is not controllable")

    # Solve the discrete-time algebraic Riccati equation
    P = solve_discrete_are(A, B, Q, R)

    # Compute the LQR gain K
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    
    # Check eigenvalues of (A - B K)
    eigenvalues = np.linalg.eigvals(A - B @ K)
    if not np.all(np.abs(eigenvalues) < 1):
        raise ValueError("Eigenvalues of (A - B K) are not all inside the unit circle.")

    return K


class LQRController(Controller):
    def __init__(self, K: np.ndarray, x_eq: np.ndarray, u_eq: np.ndarray):
        self.K = np.asarray(K)
        self.x_eq = np.asarray(x_eq)
        self.u_eq = np.asarray(u_eq)

    @classmethod
    def from_env(cls, env, Q: np.ndarray | None = None, R: np.ndarray | None = None):
        """Convenience constructor: linearize the env about hover and design K."""
        x_eq = env.hover_state()
        u_eq = np.zeros(env.action_space.shape)  # zero action = hover thrust, zero rate
        # x = [y, z, roll, phi1, phi2, ydot, zdot, rolldot, phi1dot, phi2dot]
        # 1 / (max acceptable deviation)^2)
        q = np.diag([
            1.0/(0.2**2), 1.0/(0.2**2), 1.0/(0.3**2), 
            1.0/(0.15**2), 1.0/(0.15**2), 1.0/(1.0**2), 
            1.0/(1.0**2), 1.0/(2.0**2), 1.0/(2.0**2), 
            1.0/(2.0**2)])
        # convert phi1, phi2 to q1, q2 (absolute angles)
        Tq = np.eye(5); Tq[3, 2] = 1; Tq[4, 2:4] = 1
        T = np.kron(np.eye(2), Tq)
        Q = T.T @ q @ T
        R = R if R is not None else 10*np.eye(2)
        A, B = linearize(env, x_eq, u_eq)
        K = design_lqr(A, B, Q, R)
        return cls(K, x_eq, u_eq)

    def act(self, obs, info):
        x = info["state"]
        u = self.u_eq - self.K @ (x - self.x_eq)
        return np.clip(u, -1.0, 1.0).astype(np.float32)
