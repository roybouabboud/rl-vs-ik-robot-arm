"""Jacobians for the UR5e, using MuJoCo.

Step 2 of the roadmap. Read kinematics/02_jacobians.md first,
then fill in the TODOs. Check your work with: python kinematics/check_jacobian.py
"""

import mujoco
import numpy as np

from fk import EE_SITE, ee_pose, load_ur5e


def site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, q) -> tuple[np.ndarray, np.ndarray]:
    """Return (jacp (3, nv), jacr (3, nv)) of EE_SITE at joint angles q, via mj_jacSite.

    Must not modify `data`.
    """
    saved_q = data.qpos.copy()
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = model.site(EE_SITE).id
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    data.qpos[:6] = saved_q
    mujoco.mj_forward(model, data)
    return jacp, jacr


def numerical_jacobian(model: mujoco.MjModel, data: mujoco.MjData, q, eps: float = 1e-6) -> np.ndarray:
    """Return the (3, 6) position Jacobian of EE_SITE at q, by central finite differences
    on ee_pose (no mj_jacSite allowed).
    """
    jacp = np.zeros((3, model.nv))
    for i in range(model.nv):
        dq_err = np.zeros(model.nv)
        dq_err[i] = eps
        pos_plus, _ = ee_pose(model, data, q+dq_err)
        pos_minus, _ = ee_pose(model, data, q-dq_err)
        jacp[:,i] = (pos_plus - pos_minus) / (2*eps)
    return jacp


def ee_velocity(model: mujoco.MjModel, data: mujoco.MjData, q, qdot) -> tuple[np.ndarray, np.ndarray]:
    """Predict the EE's (linear velocity (3,), angular velocity (3,)) in the world frame
    when the joints are at q and moving at qdot.
    """
    jacp, jacr = site_jacobian(model, data, q)
    xdot_pos = jacp @ qdot
    xdot_ang = jacr @ qdot
    return xdot_pos, xdot_ang


def jacobian_condition(model: mujoco.MjModel, data: mujoco.MjData, q, full: bool = True) -> float:
    """Condition number of the full 6x6 Jacobian (full=True) or the 3x6 position part (full=False)."""
    jacp, jacr = site_jacobian(model, data, q)
    if full:
        J = np.vstack([jacp, jacr])
        return np.linalg.cond(J)
    return np.linalg.cond(jacp)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    model, data = load_ur5e()
    q = np.array([0.5, -1.2, 1.0, -0.8, 0.3, 1.1])
    jacp, jacr = site_jacobian(model, data, q)
    print("jacp:\n", jacp)
    print("numerical:\n", numerical_jacobian(model, data, q))