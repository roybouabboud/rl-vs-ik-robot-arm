"""Forward kinematics for the UR5e, using MuJoCo."""

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "mujoco_menagerie" / "universal_robots_ur5e" / "ur5e.xml"
EE_SITE = "attachment_site"


def load_ur5e() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the UR5e model from MODEL_PATH and create its data."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Clone the robot models into the repo root first: "
                                "git clone https://github.com/google-deepmind/mujoco_menagerie.git")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def ee_pose(model, data, q) -> tuple[np.ndarray, np.ndarray]:
    """Return (position (3,), rotation matrix (3, 3)) of EE_SITE for joint angles q.

    Must not modify `data`.
    """
    saved_qpos = data.qpos.copy()
    data.qpos[:6] = q
    mujoco.mj_kinematics(model, data)
    site_id = model.site(EE_SITE).id
    pos = data.site_xpos[site_id].copy()
    rot = data.site_xmat[site_id].reshape(3, 3).copy()

    data.qpos[:] = saved_qpos
    mujoco.mj_kinematics(model, data)
    return pos, rot


def ee_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a [w, x, y, z] quaternion."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation_matrix.flatten())
    return quat


def sample_workspace(model: mujoco.MjModel, n: int, seed: int = 0) -> np.ndarray:
    """Return an (n, 3) array of EE positions for random joint angles within the joint limits."""
    cloud_data = mujoco.MjData(model)
    rng = np.random.default_rng(seed=seed)
    lims = model.jnt_range
    low_lims = lims[:,0]
    high_lims = lims[:, 1]
    cloud = np.zeros((n,3))
    for i in range(n):
        angs = rng.uniform(low_lims, high_lims)
        cloud_data.qpos[:6] = angs
        pos, _ = ee_pose(model, cloud_data, angs)
        cloud[i] = pos
    return cloud

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    model, data = load_ur5e()
    pos, R = ee_pose(model, data, np.zeros(6))
    print("EE position at zero pose:", pos)
    print("EE quaternion at zero pose:", ee_quat(R))
