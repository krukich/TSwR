import numpy as np
import mujoco
import mujoco.viewer

class A1Env:
    MODEL_PATH = "mujoco_menagerie/unitree_a1/scene.xml"
    OBS_VECTOR_DIM = 36
    AGENT_INPUT_DIM = 48
    TERMINAL_PENALTY = -50.0

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
        0.15, 0.35, 0.20,
        0.15, 0.35, 0.20,
        0.15, 0.35, 0.20,
        0.15, 0.35, 0.20,
    ], dtype=float)

    def __init__(
        self,
        render: bool = False,
        roll_th: float = 0.7,
        pitch_th: float = 0.7,
        z_min: float = 0.15,
        max_physics_steps: int = 2000,
        control_decimation: int = 8,
    ):
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_enabled = render
        self.viewer = None

        if control_decimation < 1:
            raise ValueError("control_decimation must be >= 1.")

        self.control_decimation = int(control_decimation)
        self.physics_timestep = float(self.model.opt.timestep)
        self.control_timestep = self.physics_timestep * self.control_decimation
        self.reset_settle_steps = 100

        self.roll_th = roll_th
        self.pitch_th = pitch_th
        self.z_min = z_min
        self.max_physics_steps = max_physics_steps

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.JOINT_NAMES
        ]

        self.previous_action = np.zeros(self.model.nu, dtype=float)
        self.step_count = 0
        self.physics_step_count = 0
        self.np_random = np.random.default_rng()

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

    def render(self):
        viewer = self.launch_viewer()
        if viewer is not None:
            viewer.sync()
        return viewer

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)

        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        self.previous_action[:] = 0.0
        self.step_count = 0
        self.physics_step_count = 0

        self.data.ctrl[:] = self.DEFAULT_POSE

        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.reset_settle_steps):
            mujoco.mj_step(self.model, self.data)

        obs_dict = self.get_observation()
        obs = self.get_agent_input_from_obs(obs_dict)
        info = self._build_info(
            terminated=False,
            truncated=False,
            termination_reason=None,
            truncation_reason=None,
        )
        return obs, info

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

    def get_obs_vector_from_obs(self, obs):
        x_t = np.concatenate([
            obs["joint_pos"],
            obs["joint_vel"],
            obs["roll_pitch"],
            obs["base_lin_vel"],
            obs["base_ang_vel"],
            obs["foot_contacts"],
        ]).astype(np.float32)

        assert x_t.shape == (self.OBS_VECTOR_DIM,)
        return x_t

    def get_obs_vector(self):
        obs = self.get_observation()
        return self.get_obs_vector_from_obs(obs)

    def get_agent_input_from_obs(self, obs):
        x_t = self.get_obs_vector_from_obs(obs)
        a_prev = obs["previous_action"]

        agent_input = np.concatenate([x_t, a_prev]).astype(np.float32)
        assert agent_input.shape == (self.AGENT_INPUT_DIM,)
        return agent_input

    def _quat_to_euler(self, quat):
        qw, qx, qy, qz = quat

        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        return roll, pitch

    def _check_terminated(self, obs):
        base_z = obs["base_pos"][2]
        roll, pitch = obs["roll_pitch"]

        if abs(roll) > self.roll_th:
            return True, "roll"
        if abs(pitch) > self.pitch_th:
            return True, "pitch"
        if base_z < self.z_min:
            return True, "base_z"

        return False, None

    def _check_truncated(self):
        if self.physics_step_count >= self.max_physics_steps:
            return True, "max_physics_steps"
        return False, None

    def is_healthy(self, obs):
        terminated, _ = self._check_terminated(obs)
        return not terminated

    def _compute_reward(self, obs_after, action, smoothness_penalty=0.0):
        vx = obs_after["base_lin_vel"][0]
        roll, pitch = obs_after["roll_pitch"]
        base_ang_vel = obs_after["base_ang_vel"]
        joint_vel = obs_after["joint_vel"]
        posture_error = roll ** 2 + pitch ** 2
        posture_scale = np.exp(-4.0 * posture_error)

        r_forward = 2.0 * max(vx, 0.0) * posture_scale
        r_posture = -1.0 * posture_error
        r_base_ang_vel = -0.05 * np.sum(base_ang_vel[:2] ** 2)
        r_action = -0.002 * np.sum(action ** 2)
        r_smooth = float(smoothness_penalty)
        r_joint_vel = -0.0005 * np.sum(joint_vel ** 2)

        reward = r_forward + r_posture + r_base_ang_vel + r_action + r_smooth + r_joint_vel

        reward_terms = {
            "r_forward": float(r_forward),
            "r_posture": float(r_posture),
            "r_base_ang_vel": float(r_base_ang_vel),
            "r_action": float(r_action),
            "r_smooth": float(r_smooth),
            "r_joint_vel": float(r_joint_vel),
            "reward_total_pre_terminal": float(reward),
        }

        return float(reward), reward_terms

    def _build_info(
        self,
        *,
        terminated,
        truncated,
        termination_reason,
        truncation_reason,
        reward_terms=None,
    ):
        info = {
            "step_count": self.step_count,
            "physics_step_count": self.physics_step_count,
            "episode_time": self.physics_step_count * self.physics_timestep,
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
        }

        if reward_terms is not None:
            info.update(reward_terms)

        return info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        prev_action = self.previous_action.copy()
        smoothness_penalty = -0.01 * np.sum((action - prev_action) ** 2)

        ctrl = self.DEFAULT_POSE + self.ACTION_SCALE * action
        self.data.ctrl[:] = ctrl

        reward = 0.0
        reward_terms = {
            "r_forward": 0.0,
            "r_posture": 0.0,
            "r_base_ang_vel": 0.0,
            "r_action": 0.0,
            "r_smooth": 0.0,
            "r_joint_vel": 0.0,
            "reward_total_pre_terminal": 0.0,
        }
        obs_after = None
        terminated = False
        termination_reason = None
        truncated = False
        truncation_reason = None

        for substep_idx in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            self.physics_step_count += 1

            obs_after = self.get_observation()
            # Smoothness is defined per policy action change, not per physics substep.
            substep_smoothness_penalty = smoothness_penalty if substep_idx == 0 else 0.0
            substep_reward, substep_terms = self._compute_reward(
                obs_after, action, substep_smoothness_penalty
            )
            reward += substep_reward

            for key, value in substep_terms.items():
                reward_terms[key] += value

            terminated, termination_reason = self._check_terminated(obs_after)
            truncated, truncation_reason = self._check_truncated()

            if terminated or truncated:
                break

        if self.viewer is not None:
            self.viewer.sync()

        self.step_count += 1
        self.previous_action = action.copy()
        obs_after = self.get_observation()
        obs = self.get_agent_input_from_obs(obs_after)

        terminal_penalty = self.TERMINAL_PENALTY if terminated else 0.0
        reward += terminal_penalty
        reward_terms["terminal_penalty"] = float(terminal_penalty)
        reward_terms["reward_total_with_terminal"] = float(reward)

        info = self._build_info(
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            truncation_reason=truncation_reason,
            reward_terms=reward_terms,
        )

        return obs, reward, terminated, truncated, info

    def stand(self):
        self.data.ctrl[:] = self.DEFAULT_POSE

        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            self.physics_step_count += 1

        if self.viewer is not None:
            self.viewer.sync()

        self.step_count += 1
        self.previous_action[:] = 0.0
        return self.get_observation()

    def get_agent_input(self):
        obs = self.get_observation()
        return self.get_agent_input_from_obs(obs)
