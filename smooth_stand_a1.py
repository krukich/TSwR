import time
import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/unitree_a1/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Более безопасная стартовая поза, чем все нули
start_ctrl = [
    0.0, 0.3, -0.6,
    0.0, 0.3, -0.6,
    0.0, 0.3, -0.6,
    0.0, 0.3, -0.6,
]

target_ctrl = [
    -0.1, 0.8, -1.5,
     0.1, 0.8, -1.5,
    -0.1, 1.0, -1.5,
     0.1, 1.0, -1.5,
]

stand_up_time = 1.0  # быстрее, чтобы не успевал развалиться


def lerp(a, b, alpha):
    return a + alpha * (b - a)


start_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - start_time
        alpha = min(t / stand_up_time, 1.0)

        for i in range(model.nu):
            data.ctrl[i] = lerp(start_ctrl[i], target_ctrl[i], alpha)

        mujoco.mj_step(model, data)
        viewer.sync()