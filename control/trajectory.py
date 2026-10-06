"""Smooth joint-space trajectories.

Step 4 (part A). Read control/04_trajectories_and_control.md first,
then fill in the TODOs. Check your work with: python control/check_control.py
"""

import numpy as np


def min_jerk(tau: float) -> tuple[float, float]:
    """Minimum-jerk time scaling. Returns (s, ds/dtau) with tau clamped into [0, 1].

    s(tau)  = 10 tau^3 - 15 tau^4 + 6 tau^5
    s'(tau) = 30 tau^2 - 60 tau^3 + 30 tau^4
    """
    # TODO 1
    raise NotImplementedError


def joint_trajectory(q_start: np.ndarray, q_goal: np.ndarray, T: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Desired (joint angles, joint velocities) at time t of a min-jerk move from q_start to q_goal lasting T seconds."""
    # TODO 2
    raise NotImplementedError


if __name__ == "__main__":
    q0, q1 = np.zeros(6), np.ones(6)
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        q, qd = joint_trajectory(q0, q1, 2.0, t)
        print(f"t={t:.1f}  q[0]={q[0]:.3f}  qd[0]={qd[0]:.3f}")
