import numpy as np
import mujoco
import mujoco.viewer


class A1Env:
    MODEL_PATH = "mujoco_menagerie/unitree_a1/scene.xml"

    JOINT_NAMES = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    DEFAULT_POSE = np.array([
        -0.11, 0.81, -1.54,
         0.11, 0.81, -1.54,
        -0.11, 1.01, -1.53,
         0.11, 1.01, -1.53,
    ], dtype=float)

    ACTION_SCALE = np.array([
        0.2, 0.65, 0.4,
        0.2, 0.65, 0.4,
        0.2, 0.65, 0.4,
        0.2, 0.65, 0.4,
    ], dtype=float)

    def __init__(
        self,
        render: bool = False,
        roll_th: float = 0.7,
        pitch_th: float = 0.7,
        z_min: float = 0.15,
        max_steps: int = 2000,
    ):
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_enabled = render
        self.viewer = None

        self.roll_th = roll_th
        self.pitch_th = pitch_th
        self.z_min = z_min
        self.max_steps = max_steps

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        ]

        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()

        self.previous_action = np.zeros(self.model.nu, dtype=float)
        self.step_count = 0

        self.foot_order = ["FR", "FL", "RR", "RL"]

        self.floor_geom_id = 0
        self.foot_geom_ids = {
            "FR": {18, 19},
            "FL": {29, 30},
            "RR": {39, 40},
            "RL": {49, 50},
        }


    def get_foot_contacts(self):
        contacts = np.zeros(4, dtype=np.float32)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            for idx, leg in enumerate(self.foot_order):
                foot_geoms = self.foot_geom_ids[leg]

                if (
                        (g1 == self.floor_geom_id and g2 in foot_geoms) or
                        (g2 == self.floor_geom_id and g1 in foot_geoms)
                ):
                    contacts[idx] = 1.0

        return contacts

    def launch_viewer(self):
        if self.render_enabled and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        self.data.ctrl[:] = self.DEFAULT_POSE
        self.previous_action[:] = 0.0
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)

        if self.render_enabled and self.viewer is None:
            self.launch_viewer()

        return self.get_observation()

    def get_observation(self):
        base_pos = self.data.qpos[0:3].copy().astype(np.float32)
        base_quat = self.data.qpos[3:7].copy().astype(np.float32)
        base_lin_vel = self.data.qvel[0:3].copy().astype(np.float32)
        base_ang_vel = self.data.qvel[3:6].copy().astype(np.float32)

        roll, pitch = self._quat_to_euler(base_quat)

        joint_pos = []
        joint_vel = []

        foot_contacts = self.get_foot_contacts()

        for jid in self.joint_ids:
            qpos_adr = self.model.jnt_qposadr[jid]
            qvel_adr = self.model.jnt_dofadr[jid]
            joint_pos.append(self.data.qpos[qpos_adr])
            joint_vel.append(self.data.qvel[qvel_adr])

        return {
            "base_pos": base_pos,
            "base_quat": base_quat,
            "roll_pitch": np.array([roll, pitch], dtype=np.float32),
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "joint_pos": np.array(joint_pos, dtype=np.float32),
            "joint_vel": np.array(joint_vel, dtype=np.float32),
            "previous_action": self.previous_action.copy().astype(np.float32),
            "foot_contacts": foot_contacts,
        }

    def get_obs_vector(self):
        obs = self.get_observation()
        x_t = np.concatenate([
            obs["joint_pos"],  # 12
            obs["joint_vel"],  # 12
            obs["roll_pitch"],  # 2
            obs["foot_contacts"],  # 4
        ]).astype(np.float32)

        return x_t

    def _quat_to_euler(self, quat):
        qw, qx, qy, qz = quat

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        return roll, pitch

    def is_healthy(self, obs):
        z = obs["base_pos"][2]
        roll, pitch = self._quat_to_euler(obs["base_quat"])

        if abs(roll) > self.roll_th:
            return False
        if abs(pitch) > self.pitch_th:
            return False
        if z < self.z_min:
            return False
        if self.step_count >= self.max_steps:
            return False

        return True

    def _compute_reward(self, obs_before, obs_after):
        forward_vel = obs_after["base_lin_vel"][0]
        reward = forward_vel
        return float(reward)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, -1.0, 1.0)

        ctrl = self.DEFAULT_POSE + self.ACTION_SCALE * action
        self.data.ctrl[:] = ctrl

        obs_before = self.get_observation()

        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        self.step_count += 1

        obs_after = self.get_observation()
        reward = self._compute_reward(obs_before, obs_after)
        done = not self.is_healthy(obs_after)

        self.previous_action = action.copy()

        info = {
            "step_count": self.step_count,
        }

        return obs_after, reward, done, info

    def stand(self):
        self.data.ctrl[:] = self.DEFAULT_POSE
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        self.step_count += 1
        return self.get_observation()