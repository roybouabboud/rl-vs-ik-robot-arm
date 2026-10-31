"""Torque control for the UR5e: PD + gravity compensation along a min-jerk trajectory."""

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kinematics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fk import MODEL_PATH  # noqa: E402
from trajectory import joint_trajectory  # noqa: E402

# Default gains per joint: big shoulder/elbow joints, then the 3 light wrist joints.
KP = np.array([400.0, 400.0, 400.0, 100.0, 100.0, 100.0])  # N·m / rad
KD = np.array([40.0, 40.0, 40.0, 10.0, 10.0, 10.0])        # N·m / (rad/s)


def make_torque_model(with_target: bool = True) -> mujoco.MjModel:
    """The UR5e scene with every actuator switched to a torque motor (ctrl = N·m, clipped to the
    real UR5e limits), plus an optional red mocap sphere to mark targets.
    """
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH.parent / "scene.xml"))
    for act in spec.actuators:
        limit = np.array(act.forcerange).copy()
        act.set_to_motor()            # gain 1, no bias: ctrl is applied directly as torque
        act.ctrllimited = 1
        act.ctrlrange = limit
    if with_target:
        target = spec.worldbody.add_body(name="target", mocap=True, pos=[0, 0, -1])
        target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0],
                        rgba=[1, 0.2, 0.2, 0.6], contype=0, conaffinity=0)
    return spec.compile()


def gravity_compensation(data: mujoco.MjData) -> np.ndarray:
    """The 6 joint torques that cancel gravity (and Coriolis) at the current state, as a copy."""
    return data.qfrc_bias[:6].copy()


def pd_gravity_torque(data: mujoco.MjData, q_d: np.ndarray, qd_d: np.ndarray,
                      kp: np.ndarray = KP, kd: np.ndarray = KD, gravity: bool = True) -> np.ndarray:
    """tau = kp (q_d - q) + kd (qd_d - qd) [+ gravity compensation]."""
    tor = kp*(q_d - data.qpos[:6]) + kd*(qd_d - data.qvel[:6])
    if gravity:  tor += gravity_compensation(data)
    return tor


def move_to(model: mujoco.MjModel, data: mujoco.MjData, q_goal: np.ndarray, duration: float = 2.0,
            kp: np.ndarray = KP, kd: np.ndarray = KD, gravity: bool = True,
            settle: float = 1.0, viewer=None) -> float:
    """Follow a min-jerk trajectory from the current joint angles to q_goal over `duration` seconds,
    then hold for `settle` more seconds, simulating with mj_step1 / set ctrl / mj_step2.

    If `viewer` is given, sync it and sleep one timestep per step (real time).
    Returns the largest joint tracking error max|q_d - q| [rad] seen along the way.
    """
    q_start = data.qpos[:model.nv].copy()
    t0 = data.time
    worst = 0.0
    step = int(round((duration + settle) / model.opt.timestep))
    for _ in range(step):
        mujoco.mj_step1(model, data)
        t = data.time - t0
        q_des, dq_des = joint_trajectory(q_start, q_goal, duration, t)
        q = data.qpos[:6]
        if np.linalg.norm(q_des - q) > worst: worst = np.linalg.norm(q_des - q)
        data.ctrl[:] = pd_gravity_torque(data, q_des, dq_des, kp, kd, gravity=gravity)
        mujoco.mj_step2(model, data)
        if viewer is not None: 
            viewer.sync()
            time.sleep(model.opt.timestep)
    return worst


if __name__ == "__main__":
    model = make_torque_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    q_goal = np.array([-2.89142, -1.693179, 1.667078, -1.513458, -1.974193, 0.0])
    for g in (True, False):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        err = move_to(model, data, q_goal, gravity=g)
        print(f"gravity={g}: max tracking error {err:.4f} rad, final error {np.abs(q_goal - data.qpos[:6]).max():.5f} rad")
