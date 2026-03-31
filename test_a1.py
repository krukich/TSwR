import time
import math
import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/unitree_a1/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Посмотрим, какой тип управления у актуаторов
print("nu =", model.nu)
print("First actuator name:",
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 0))

start = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - start

        # Сначала все управления в 0
        for i in range(model.nu):
            data.ctrl[i] = 0.0

        # Качаем только FR_hip (actuator 0)
        data.ctrl[0] = 0.2 * math.sin(2.0 * t)

        mujoco.mj_step(model, data)
        viewer.sync()