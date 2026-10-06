"""Position-only inverse kinematics for the UR5e.

Step 3 of the roadmap. Read kinematics/03_inverse_kinematics.md first,
then fill in the TODOs. Check your work with: python kinematics/check_ik.py
"""

import mujoco
import numpy as np

from fk import ee_pose, load_ur5e
from jacobian import site_jacobian


def clamp_to_limits(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    """Return q with each joint angle clipped into model.jnt_range."""
    # TODO 1
    raise NotImplementedError


def limit_step(dq: np.ndarray, max_step: float | None) -> np.ndarray:
    """Return dq scaled down so its length is at most max_step (unchanged if already shorter,
    or if max_step is None).
    """
    # TODO 2
    raise NotImplementedError


def ik_pinv(model: mujoco.MjModel, data: mujoco.MjData, target, q0,
            max_iters: int = 200, tol: float = 1e-4, max_step: float | None = 0.5,
            ) -> tuple[np.ndarray, bool, int]:
    """Solve for joint angles that put EE_SITE at `target` (3,), starting from q0,
    using the pseudo-inverse of the position Jacobian.

    Returns (q, success, iterations_used). success means |target - ee_pos(q)| < tol.
    Must not modify `data`.
    """
    # TODO 3
    raise NotImplementedError


def ik_dls(model: mujoco.MjModel, data: mujoco.MjData, target, q0,
           max_iters: int = 200, tol: float = 1e-4, max_step: float | None = 0.5,
           damping: float = 0.05) -> tuple[np.ndarray, bool, int]:
    """Same as ik_pinv, but using damped least squares with the given damping (lambda)."""
    # TODO 4
    raise NotImplementedError


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    model, data = load_ur5e()
    q0 = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])  # the model's "home" keyframe
    target = np.array([0.4, 0.2, 0.5])
    for solver in (ik_pinv, ik_dls):
        q, ok, iters = solver(model, data, target, q0)
        print(f"{solver.__name__}: success={ok} in {iters} iters, q={q}, hand at {ee_pose(model, data, q)[0]}")
