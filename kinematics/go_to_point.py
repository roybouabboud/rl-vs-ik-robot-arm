"""Type a point, the arm goes there.

Run from the repo root:  python kinematics/go_to_point.py
Then type `x y z` in this terminal (e.g. `0.4 0.2 0.5`) and press Enter. `q` quits.
Several points at once, visited in order: `0.4 0.2 0.5; 0.3 -0.4 0.2; -0.5 0.1 0.1`.
"""

import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fk import EE_SITE, MODEL_PATH, ee_pose  # noqa: E402
from ik import ik_dls  # noqa: E402


def build_model() -> mujoco.MjModel:
    """The UR5e scene (floor + lights) plus a red, non-colliding 'mocap' sphere to mark the target.

    A mocap body is moved by setting data.mocap_pos directly; physics never pushes it around.
    """
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH.parent / "scene.xml"))
    target = spec.worldbody.add_body(name="target", mocap=True, pos=[0, 0, -1])
    target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0],
                    rgba=[1, 0.2, 0.2, 0.6], contype=0, conaffinity=0)
    return spec.compile()


def read_targets(queue: list, stop: threading.Event):
    """Background thread: read lines of one or more 'x y z' points from the terminal."""
    while not stop.is_set():
        try:
            line = input("target x y z > ").strip()
        except EOFError:
            break
        if line.lower() in ("q", "quit", "exit"):
            stop.set()
            break
        try:
            values = np.array([float(v) for v in line.replace(",", " ").replace(";", " ").split()])
            assert values.size > 0 and values.size % 3 == 0
            queue.extend(values.reshape(-1, 3))
        except (ValueError, AssertionError):
            print("  please type points as x y z, e.g. 0.4 0.2 0.5  or  0.4 0.2 0.5; 0.3 -0.4 0.2")


def main():
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[:] = data.qpos[:6]
    site_id = model.site(EE_SITE).id

    print("Hand starts at", np.round(ee_pose(model, data, data.qpos[:6].copy())[0], 3))
    queue, stop = [], threading.Event()
    threading.Thread(target=read_targets, args=(queue, stop), daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        target, report_at = None, None
        while viewer.is_running() and not stop.is_set():
            if queue and report_at is None:  # wait for the arm to settle before the next point
                target = queue.pop(0)
                data.mocap_pos[0] = target
                q, ok, iters = ik_dls(model, data, target, data.qpos[:6].copy())
                if ok:
                    print(f"  IK solved in {iters} iters. Moving...")
                else:
                    print(f"  IK failed after {iters} iters (out of reach?). Moving as close as it got.")
                data.ctrl[:] = q  # position servos: ctrl is the target joint angle
                report_at = time.time() + 2.0

            mujoco.mj_step(model, data)
            viewer.sync()

            if report_at and time.time() > report_at:
                err = np.linalg.norm(target - data.site_xpos[site_id])
                print(f"  settled: hand is {err * 1000:.1f} mm from the target")
                report_at = None
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
