"""Inverse kinematics for the UR5e on MuJoCo's Jacobian: position-only (pseudo-inverse and
damped least squares) and full pose (position + orientation, with random restarts)."""

import mujoco
import numpy as np

from fk import ee_pose, ee_quat, load_ur5e
from jacobian import site_jacobian


def clamp_to_limits(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    """Return q with each joint angle clipped into model.jnt_range."""
    low_lims = model.jnt_range[:,0]
    high_lims = model.jnt_range[:,1]
    clip_q = np.clip(q, low_lims, high_lims)
    return clip_q


def limit_step(dq: np.ndarray, max_step: float | None) -> np.ndarray:
    """Return dq scaled down so its length is at most max_step (unchanged if already shorter,
    or if max_step is None).
    """
    length = np.linalg.norm(dq)
    if max_step is None: return dq
    if length > max_step:
        norm = dq / length
        dq = max_step * norm
        return dq
    return dq


def ik_pinv(model: mujoco.MjModel, data: mujoco.MjData, target, q0,
            max_iters: int = 200, tol: float = 1e-4, max_step: float | None = 0.5,
            ) -> tuple[np.ndarray, bool, int]:
    """Solve for joint angles that put EE_SITE at `target` (3,), starting from q0,
    using the pseudo-inverse of the position Jacobian.

    Returns (q, success, iterations_used). success means |target - ee_pos(q)| < tol.
    Must not modify `data`.
    """
    q = q0.copy()
    for i in range(1, max_iters+2):
        pos, rot = ee_pose(model, data, q)
        err = target - pos
        norm_err = np.linalg.norm(err)
        if norm_err < tol:
            return q, True, i
        if i == max_iters+1: break
        jacp, jacr = site_jacobian(model, data, q)
        dq = np.linalg.pinv(jacp) @ err
        dq = limit_step(dq, max_step)
        q = clamp_to_limits(model, q+dq)
    return q, False, max_iters


def ik_dls(model: mujoco.MjModel, data: mujoco.MjData, target, q0,
           max_iters: int = 200, tol: float = 1e-4, max_step: float | None = 0.5,
           damping: float = 0.05) -> tuple[np.ndarray, bool, int]:
    """Same as ik_pinv, but using damped least squares with the given damping (lambda)."""
    q = q0.copy()
    for i in range(1, max_iters+2):
        pos, rot = ee_pose(model, data, q)
        err = target - pos
        norm_err = np.linalg.norm(err)
        if norm_err < tol:
            return q, True, i
        if i == max_iters+1: break
        jacp, jacr = site_jacobian(model, data, q)
        dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping*damping * np.eye(3), err)
        dq = limit_step(dq, max_step)
        q = clamp_to_limits(model, q+dq)
    return q, False, max_iters


# --- Full pose (position + orientation) -------------------------------------------

# Flange pointing straight down (tool z axis = world -z). Same as the "home" keyframe's orientation.
R_DOWN = np.diag([1.0, -1.0, -1.0])

def rotation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """World-frame 3-vector (axis * angle) that rotates R_current onto R_target."""
    err = np.zeros(3)
    w_cur = ee_quat(R_current)
    w_target = ee_quat(R_target)
    mujoco.mju_subQuat(err, w_target, w_cur)
    err_world = R_current @ err
    return err_world

def pose_error(model: mujoco.MjModel, data: mujoco.MjData, q, target_pos, target_R) -> np.ndarray:
    """6-vector [position error (3), rotation error (3)] of the EE at joint angles q.

    Must not modify `data`.
    """
    cur_pos, cur_rot = ee_pose(model, data, q)
    pos_err = target_pos - cur_pos
    rot_err = rotation_error(cur_rot, target_R)
    return np.concatenate((pos_err, rot_err))


def ik_pose(model: mujoco.MjModel, data: mujoco.MjData, target_pos, target_R, q0,
            max_iters: int = 200, pos_tol: float = 1e-4, rot_tol: float = 1e-3,
            max_step: float | None = 0.5, damping: float = 0.05) -> tuple[np.ndarray, bool, int]:
    """DLS IK on the full 6x6 Jacobian. Success means position error < pos_tol [m]
    and rotation error < rot_tol [rad].

    Returns (q, success, iterations_used). Must not modify `data`.
    """
    q = q0.copy()
    for i in range(1, max_iters+2):
        pose_err = pose_error(model, data, q, target_pos, target_R)
        pos_err = pose_err[:3]
        rot_err = pose_err[3:]
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            return q, True, i
        if i == max_iters+1: break
        jacp, jacr = site_jacobian(model, data, q)
        J = np.vstack([jacp, jacr])
        dq = J.T @ np.linalg.solve(J @ J.T + damping*damping * np.eye(6), pose_err)
        dq = limit_step(dq, max_step)
        q = clamp_to_limits(model, q+dq)
    return q, False, max_iters


def ik_pose_restarts(model: mujoco.MjModel, data: mujoco.MjData, target_pos, target_R, q0,
                     n_restarts: int = 10, seed: int = 0, **ik_kwargs) -> tuple[np.ndarray, bool, int]:
    """Try ik_pose from q0, then from up to n_restarts random joint configurations
    (within the joint limits, drawn from np.random.default_rng(seed)).
    Extra keyword arguments are passed through to ik_pose.

    Returns (q, success, restarts_used): restarts_used is 0 if q0 itself worked.
    """
    rng = np.random.default_rng(seed=seed)
    q = q0
    for i in range(n_restarts+1):
        q_new, test, iters = ik_pose(model, data, target_pos, target_R, q, **ik_kwargs)
        if test:
            return q_new, test, i
        q = rng.uniform(model.jnt_range[:,0], model.jnt_range[:,1])
    return q_new, test, i


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    model, data = load_ur5e()
    q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])  # the model's "home" keyframe
    target = np.array([0.4, 0.2, 0.5])
    for solver in (ik_pinv, ik_dls):
        q, ok, iters = solver(model, data, target, q0)
        print(f"{solver.__name__}: success={ok} in {iters} iters, q={q}, hand at {ee_pose(model, data, q)[0]}")

    q, ok, n = ik_pose_restarts(model, data, np.array([0.4, 0.2, 0.3]), R_DOWN, q0)
    pos, R = ee_pose(model, data, q)
    print(f"ik_pose_restarts (pointing down): success={ok} after {n} restarts, hand at {pos}")
    print("hand R=\n", R)
