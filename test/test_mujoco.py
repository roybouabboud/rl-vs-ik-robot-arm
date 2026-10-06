import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_string("""
<mujoco>
    <worldbody>
        <light pos="0 0 3"/>

        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom type="box"
                  size="0.2 0.2 0.2"
                  rgba="0.2 0.6 1 1"
                  mass="1"/>
        </body>

        <geom type="plane"
              size="5 5 0.1"
              rgba="0.5 0.5 0.5 1"/>
    </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print(mujoco.__file__)