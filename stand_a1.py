import mujoco
import mujoco.viewer

MODEL_PATH = "mujoco_menagerie/unitree_a1/scene.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Порядок actuator'ов:
# 0  FR_hip
# 1  FR_thigh
# 2  FR_calf
# 3  FL_hip
# 4  FL_thigh
# 5  FL_calf
# 6  RR_hip
# 7  RR_thigh
# 8  RR_calf
# 9  RL_hip
# 10 RL_thigh
# 11 RL_calf

stand_ctrl = [
    -0.1, 0.8, -1.5,
     0.1, 0.8, -1.5,
    -0.1, 1.0, -1.5,
     0.1, 1.0, -1.5,
]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for i in range(model.nu):
            data.ctrl[i] = stand_ctrl[i]

        mujoco.mj_step(model, data)
        viewer.sync()