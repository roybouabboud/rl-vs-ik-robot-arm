"""Smooth joint-space trajectories."""

import numpy as np


def min_jerk(tau: float) -> tuple[float, float]:
    """Minimum-jerk time scaling. Returns (s, ds/dtau) with tau clamped into [0, 1].

    s(tau)  = 10 tau^3 - 15 tau^4 + 6 tau^5
    s'(tau) = 30 tau^2 - 60 tau^3 + 30 tau^4
    """
    t = np.clip(tau, 0, 1)
    s = 10*t**3 - 15*t**4 + 6*t**5
    ds = 30*t**2 - 60*t**3 + 30*t**4
    return s, ds


def joint_trajectory(q_start: np.ndarray, q_goal: np.ndarray, T: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Desired (joint angles, joint velocities) at time t of a min-jerk move from q_start to q_goal lasting T seconds."""
    s, ds = min_jerk((t/T))
    q_des = q_start + (q_goal - q_start)*s
    dq_des = (q_goal - q_start)*(ds/T)
    return q_des, dq_des


if __name__ == "__main__":
    q0, q1 = np.zeros(6), np.ones(6)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        q, qd = joint_trajectory(q0, q1, 2.0, t)
        print(f"t={t:.1f}  q[0]={q[0]:.3f}  qd[0]={qd[0]:.3f}")
