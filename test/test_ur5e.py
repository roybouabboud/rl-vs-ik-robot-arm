import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
)
data = mujoco.MjData(model)

target = [1, 0.2, -0.3, -0.7, 1.5, 2]
kp = 15
kd = 40
ki = 8

cur_err = np.zeros(6)
prev_err = np.zeros(6)
integral_err = np.zeros(6)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # data.qvel[0] = 0.1
        # data.qpos[1] = -1
        # data.qpos[2] = -2
        # data.qpos[3] = 0.785
        # data.qpos[4] = 0.785*2
        # data.qvel[5] = 0.785*2

        q = data.qpos[:6]
        dq = data.qvel[:6]

        cur_err = target - q
        dcur_err = -dq
        integral_err += (cur_err + prev_err)*model.opt.timestep*0.5

        P = kp*cur_err
        I = ki*integral_err
        D = kd*dcur_err

        data.ctrl[:6] = P + I + D
        prev_err = cur_err

        mujoco.mj_step(model, data)
        viewer.sync()