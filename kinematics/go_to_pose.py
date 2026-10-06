"""Type a point, the arm goes there pointing straight down.

Run from the repo root:  python kinematics/go_to_pose.py
Then type `x y z` in this terminal (e.g. `0.4 0.2 0.3`) and press Enter. `q` quits.
Needs ik_pose_restarts in pose_ik.py done.
"""

import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fk import EE_SITE  # noqa: E402
from go_to_point import build_model, read_targets  # noqa: E402
from pose_ik import R_DOWN, ik_pose_restarts, rotation_error  # noqa: E402


def main():
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[:] = data.qpos[:6]
    site_id = model.site(EE_SITE).id

    queue, stop = [], threading.Event()
    threading.Thread(target=read_targets, args=(queue, stop), daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        target, report_at = None, None
        while viewer.is_running() and not stop.is_set():
            if queue:
                target = queue.pop(0)
                data.mocap_pos[0] = target
                q, ok, restarts = ik_pose_restarts(model, data, target, R_DOWN, data.qpos[:6].copy())
                if ok:
                    print(f"  IK solved ({restarts} restarts). Moving...")
                else:
                    print("  IK failed: can't reach that point pointing down. Moving as close as it got.")
                data.ctrl[:] = q
                report_at = time.time() + 2.0

            mujoco.mj_step(model, data)
            viewer.sync()

            if report_at and time.time() > report_at:
                pos_err = np.linalg.norm(target - data.site_xpos[site_id])
                rot_err = np.linalg.norm(rotation_error(data.site_xmat[site_id].reshape(3, 3), R_DOWN))
                print(f"  settled: {pos_err * 1000:.1f} mm from the target, {np.degrees(rot_err):.1f} deg from pointing down")
                report_at = None
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
