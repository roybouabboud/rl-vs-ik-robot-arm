"""Type a point: IK plans the joint angles, a min-jerk trajectory gets there smoothly,
and your PD + gravity-compensation torque controller holds it precisely.

Run from the repo root:  python control/go_smooth.py
Type `x y z` (or several: `0.4 0.2 0.5; 0.3 -0.4 0.2`) and press Enter. `q` quits.
Needs step 4's TODOs done (and ik_dls from step 3).
"""

import sys
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kinematics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from controller import make_torque_model, move_to  # noqa: E402
from fk import EE_SITE  # noqa: E402
from go_to_point import read_targets  # noqa: E402
from ik import ik_dls  # noqa: E402


def main():
    model = make_torque_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    site_id = model.site(EE_SITE).id

    queue, stop = [], threading.Event()
    threading.Thread(target=read_targets, args=(queue, stop), daemon=True).start()

    q_hold = data.qpos[:6].copy()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not stop.is_set():
            if queue:
                target = queue.pop(0)
                data.mocap_pos[0] = target
                q_goal, ok, iters = ik_dls(model, data, target, data.qpos[:6].copy())
                print(f"  IK {'solved' if ok else 'FAILED (out of reach?)'} in {iters} iters. Moving...")
                track = move_to(model, data, q_goal, duration=2.0, viewer=viewer)
                q_hold = q_goal
                err = np.linalg.norm(target - data.site_xpos[site_id])
                print(f"  settled: hand is {err * 1000:.2f} mm from the target (max tracking error {np.degrees(track):.2f} deg)")
            else:
                # idle: keep holding the last goal (the controller has to keep running, or gravity wins)
                move_to(model, data, q_hold, duration=0.05, settle=0.0, viewer=viewer)


if __name__ == "__main__":
    main()
