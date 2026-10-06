"""Full-pose (position + orientation) inverse kinematics for the UR5e.

Step 3b of the roadmap. Read kinematics/03b_pose_ik.md first,
then fill in the TODOs. Check your work with: python kinematics/check_pose_ik.py
"""

import mujoco
import numpy as np

from fk import ee_pose, ee_quat, load_ur5e
from ik import clamp_to_limits, limit_step
from jacobian import site_jacobian

# Flange pointing straight down (tool z axis = world -z). Same as the "home" keyframe's orientation.
R_DOWN = np.diag([1.0, -1.0, -1.0])


def rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """World-frame 3-vector (axis * angle) that rotates R_current onto R_target."""
    # TODO 1
    raise NotImplementedError


def pose_error(model: mujoco.MjModel, data: mujoco.MjData, q, target_pos, target_R) -> np.ndarray:
    """6-vector [position error (3), rotation error (3)] of the EE at joint angles q.

    Must not modify `data`.
    """
    # TODO 2
    raise NotImplementedError


def ik_pose(model: mujoco.MjModel, data: mujoco.MjData, target_pos, target_R, q0,
            max_iters: int = 200, pos_tol: float = 1e-4, rot_tol: float = 1e-3,
            max_step: float | None = 0.5, damping: float = 0.05) -> tuple[np.ndarray, bool, int]:
    """DLS IK on the full 6x6 Jacobian. Success means position error < pos_tol [m]
    and rotation error < rot_tol [rad].

    Returns (q, success, iterations_used). Must not modify `data`.
    """
    # TODO 3
    raise NotImplementedError


def ik_pose_restarts(model: mujoco.MjModel, data: mujoco.MjData, target_pos, target_R, q0,
                     n_restarts: int = 10, seed: int = 0, **ik_kwargs) -> tuple[np.ndarray, bool, int]:
    """Try ik_pose from q0, then from up to n_restarts random joint configurations
    (within the joint limits, drawn from np.random.default_rng(seed)).
    Extra keyword arguments are passed through to ik_pose.

    Returns (q, success, restarts_used): restarts_used is 0 if q0 itself worked.
    """
    # TODO 4
    raise NotImplementedError


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    model, data = load_ur5e()
    home = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
    target = np.array([0.4, 0.2, 0.3])
    q, ok, n = ik_pose_restarts(model, data, target, R_DOWN, home)
    pos, R = ee_pose(model, data, q)
    print(f"success={ok} after {n} restarts\nq={q}\nhand at {pos}\nhand R=\n{R}")
