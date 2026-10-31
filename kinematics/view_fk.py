"""Drag the UR5e's actuator sliders and watch the ee_pose() output live.

Run from the repo root:  python kinematics/view_fk.py
Sliders: right-hand panel -> Control.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fk import ee_pose, load_ur5e  # noqa: E402

model, data = load_ur5e()
np.set_printoptions(precision=3, suppress=True)
last_print = 0.0

with mujoco.viewer.launch_passive(model, data, show_right_ui=True) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        if time.time() - last_print > 0.5:
            pos, _ = ee_pose(model, data, data.qpos[:6].copy())
            print("q =", data.qpos[:6], " EE =", pos)
            last_print = time.time()
        time.sleep(model.opt.timestep)
