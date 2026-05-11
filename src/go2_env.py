import math
import os

import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        device="cuda:0",
        add_camera=False,
    ):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.base_num_obs = obs_cfg.get("base_num_obs", obs_cfg["num_obs"])
        self.privileged_raw_dim = obs_cfg.get("privileged_raw_dim", 5)
        self.privileged_encoder_dim = obs_cfg.get("privileged_encoder_dim", 6)
        computed_num_obs = self.base_num_obs + self.privileged_encoder_dim
        computed_num_privileged_obs = computed_num_obs + self.privileged_raw_dim
        self.num_obs = obs_cfg.get("num_obs", computed_num_obs)
        if self.num_obs == self.base_num_obs:
            self.num_obs = computed_num_obs
        self.num_privileged_obs = obs_cfg.get("num_privileged_obs", computed_num_privileged_obs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = env_cfg["simulate_action_latency"]

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]

        self.reward_scales = dict(reward_cfg["reward_scales"])

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=min(num_envs, env_cfg.get("n_rendered_envs", num_envs)),
                show_world_frame=False,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_links_info=True,
                batch_dofs_info=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        ground_friction = self.env_cfg.get("ground_friction", None)

        if ground_friction is not None:
            print(f"[Go2Env] ground_friction = {ground_friction}")
            self.ground = self.scene.add_entity(
                gs.morphs.Plane(),
                material=gs.materials.Rigid(friction=float(ground_friction)),
            )
        else:
            plane_path = env_cfg.get("plane_path", "urdf/plane/plane.urdf")
            if os.path.exists(plane_path):
                self.ground = self.scene.add_entity(gs.morphs.URDF(file=plane_path, fixed=True))
            else:
                self.ground = self.scene.add_entity(gs.morphs.Plane())
        self.ground_friction_value = float(ground_friction) if ground_friction is not None else 1.0

        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg["asset_path"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            )
        )

        self.cam_0 = None
        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 3.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )

        self.scene.build(
            n_envs=num_envs,
            env_spacing=(1.0, 1.0),
        )

        self._apply_contact_friction_overrides(ground_friction)
        self._setup_friction_randomization(ground_friction)

        self.payload_link = self._find_payload_link()
        self.payload_ls_idx_local = [self.payload_link.idx_local]
        self.payload_link_mass = float(self.payload_link.get_mass())
        self._log_payload_setup()

        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local
            for name in self.env_cfg["dof_names"]
        ]

        print("============================================================")
        print("GO2 MOTOR DOFS")
        print("============================================================")
        for name, idx in zip(self.env_cfg["dof_names"], self.motor_dofs):
            print(f"{name:20s} -> {idx}")
        print("motor_dofs =", self.motor_dofs)
        print("unique =", sorted(set(self.motor_dofs)))
        print("============================================================")

        if len(set(self.motor_dofs)) != self.num_actions:
            raise RuntimeError(f"Broken motor DOF mapping: {self.motor_dofs}")

        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.num_actions,
            self.motor_dofs,
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.num_actions,
            self.motor_dofs,
        )

        if "force_low" in self.env_cfg and "force_high" in self.env_cfg:
            self.robot.set_dofs_force_range(
                [self.env_cfg["force_low"]] * self.num_actions,
                [self.env_cfg["force_high"]] * self.num_actions,
                self.motor_dofs,
            )

        self.reward_functions = {}
        self.episode_sums = {}

        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,),
                device=self.device,
                dtype=gs.tc_float,
            )

        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)

        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.privileged_obs_buf = torch.zeros(
            (self.num_envs, self.num_privileged_obs),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.reset_buf = torch.ones(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_int,
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_int,
        )

        self.commands = torch.zeros(
            (self.num_envs, self.num_commands),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.actions = torch.zeros(
            (self.num_envs, self.num_actions),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_actions = torch.zeros_like(self.actions)

        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        self.base_pos = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.base_euler = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["dof_names"]
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.jump_toggled_buf = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.jump_target_height = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.ground_friction_buf = torch.full(
            (self.num_envs, 1),
            self.default_ground_friction,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.payload_mass_shift = torch.zeros(
            (self.num_envs, 1),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.payload_pos_local = torch.zeros(
            (self.num_envs, 1, 3),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.payload_com_shift = torch.zeros(
            (self.num_envs, 1, 3),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.extras = {}

        self.reset()

    def _apply_contact_friction_overrides(self, ground_friction):
        foot_friction = self.env_cfg.get("foot_friction", ground_friction)
        self.contact_links = self._get_robot_contact_links()
        self.contact_link_ls_idx_local = [link.idx_local for link in self.contact_links]
        self.default_ground_friction = float(ground_friction) if ground_friction is not None else 1.0
        self.default_foot_friction = float(foot_friction) if foot_friction is not None else 1.0

        if foot_friction is None or not self.contact_links:
            if not self.contact_links:
                print("[Go2Env] warning: no contact links found for friction override")
            return

        foot_friction = float(foot_friction)
        for link in self.contact_links:
            link.set_friction(foot_friction)

        link_names = ", ".join(link.name for link in self.contact_links)
        print(f"[Go2Env] foot_friction = {foot_friction} on {link_names}")

    def _get_robot_contact_links(self):
        contact_link_names = self.env_cfg.get("contact_link_names")
        foot_link_names = self.env_cfg.get("foot_link_names")

        if contact_link_names is not None:
            return [self.robot.get_link(name) for name in contact_link_names]
        if foot_link_names is not None:
            return [self.robot.get_link(name) for name in foot_link_names]

        foot_links = [link for link in self.robot.links if link.name.endswith("_foot")]
        if foot_links:
            return foot_links

        return [link for link in self.robot.links if link.name.endswith("_calf")]

    def _setup_friction_randomization(self, ground_friction):
        self.ground_link_ls_idx_local = [link.idx_local for link in self.ground.links]
        self.default_ground_friction = float(ground_friction) if ground_friction is not None else 1.0
        self.ground_friction_range = None if ground_friction is not None else self.env_cfg.get("ground_friction_range")

        if self.ground_friction_range is not None:
            low, high = self._get_cfg_range("ground_friction_range")
            print(f"[Go2Env] ground_friction_range = ({low}, {high})")

    def _sample_ground_friction(self, envs_idx):
        if len(envs_idx) == 0:
            return

        if self.ground_friction_range is not None:
            low, high = self._get_cfg_range("ground_friction_range")
            friction = gs_rand_float(low, high, (len(envs_idx),), self.device)
        else:
            friction = torch.full(
                (len(envs_idx),),
                self.default_ground_friction,
                device=self.device,
                dtype=gs.tc_float,
            )

        friction = torch.clamp(friction, 1.0e-2, 5.0)
        self.ground_friction_buf[envs_idx, 0] = friction

        if self.ground_link_ls_idx_local:
            ground_ratio = (friction / max(self.default_ground_friction, 1.0e-6)).unsqueeze(-1)
            ground_ratio = ground_ratio.repeat(1, len(self.ground_link_ls_idx_local))
            self.ground.set_friction_ratio(
                ground_ratio,
                self.ground_link_ls_idx_local,
                envs_idx,
            )

        if self.contact_link_ls_idx_local:
            foot_ratio = (friction / max(self.default_foot_friction, 1.0e-6)).unsqueeze(-1)
            foot_ratio = foot_ratio.repeat(1, len(self.contact_link_ls_idx_local))
            self.robot.set_friction_ratio(
                foot_ratio,
                self.contact_link_ls_idx_local,
                envs_idx,
            )

    def _find_payload_link(self):
        preferred_name = self.env_cfg.get("payload_link_name", None)

        if preferred_name is not None:
            try:
                link = self.robot.get_link(preferred_name)
                print(f"[Go2Env] payload link = {link.name}")
                return link
            except Exception:
                print(f"[Go2Env] warning: payload_link_name='{preferred_name}' not found")

        for name in ["base_link", "base", "trunk"]:
            try:
                link = self.robot.get_link(name)
                print(f"[Go2Env] payload link = {link.name}")
                return link
            except Exception:
                pass

        print("[Go2Env] warning: using robot.base_link as payload link")
        print(f"[Go2Env] robot.base_link = {self.robot.base_link.name}")
        return self.robot.base_link

    def _get_cfg_range(self, key, default=(0.0, 0.0)):
        values = self.env_cfg.get(key, default)
        if len(values) != 2:
            raise ValueError(f"{key} must have exactly two values, got {values}")
        return float(values[0]), float(values[1])

    def _get_cfg_range_with_fallback(self, primary_key, fallback_key):
        if primary_key in self.env_cfg:
            return self._get_cfg_range(primary_key)
        return self._get_cfg_range(fallback_key)

    def _log_payload_setup(self):
        mass_range = self._get_cfg_range("payload_mass_range")
        pos_x_range = self._get_cfg_range("payload_pos_x_range")
        pos_y_range = self._get_cfg_range("payload_pos_y_range")
        pos_z_range = self._get_cfg_range("payload_pos_z_range")

        if max(
            abs(mass_range[0]),
            abs(mass_range[1]),
            abs(pos_x_range[0]),
            abs(pos_x_range[1]),
            abs(pos_y_range[0]),
            abs(pos_y_range[1]),
            abs(pos_z_range[0]),
            abs(pos_z_range[1]),
        ) <= 0.0:
            return

        print(
            "[Go2Env] payload carrier="
            f"{self.payload_link.name} "
            f"carrier_mass={self.payload_link_mass:.3f}kg "
            f"mass_range={mass_range} "
            f"pos_x_range={pos_x_range} "
            f"pos_y_range={pos_y_range} "
            f"pos_z_range={pos_z_range}"
        )

    def _sample_payload(self, envs_idx):
        if len(envs_idx) == 0:
            return

        num_envs = len(envs_idx)
        mass_low, mass_high = self._get_cfg_range("payload_mass_range")
        pos_x_low, pos_x_high = self._get_cfg_range("payload_pos_x_range")
        pos_y_low, pos_y_high = self._get_cfg_range("payload_pos_y_range")
        pos_z_low, pos_z_high = self._get_cfg_range("payload_pos_z_range")

        self.payload_mass_shift[envs_idx, 0] = gs_rand_float(
            mass_low,
            mass_high,
            (num_envs,),
            self.device,
        )
        self.payload_pos_local[envs_idx, 0, 0] = gs_rand_float(
            pos_x_low,
            pos_x_high,
            (num_envs,),
            self.device,
        )
        self.payload_pos_local[envs_idx, 0, 1] = gs_rand_float(
            pos_y_low,
            pos_y_high,
            (num_envs,),
            self.device,
        )
        self.payload_pos_local[envs_idx, 0, 2] = gs_rand_float(
            pos_z_low,
            pos_z_high,
            (num_envs,),
            self.device,
        )

        total_link_mass = self.payload_link_mass + self.payload_mass_shift[envs_idx, 0]
        payload_ratio = torch.where(
            total_link_mass > 1.0e-6,
            self.payload_mass_shift[envs_idx, 0] / total_link_mass,
            torch.zeros_like(total_link_mass),
        )

        self.payload_com_shift[envs_idx, 0] = (
            self.payload_pos_local[envs_idx, 0] * payload_ratio.unsqueeze(-1)
        )

        self.robot.set_mass_shift(
            self.payload_mass_shift[envs_idx],
            self.payload_ls_idx_local,
            envs_idx,
        )
        self.robot.set_COM_shift(
            self.payload_com_shift[envs_idx],
            self.payload_ls_idx_local,
            envs_idx,
        )

    def _normalize_range(self, values, low, high):
        denom = max(float(high) - float(low), 1.0e-6)
        return 2.0 * ((values - float(low)) / denom) - 1.0

    def _get_privileged_raw(self):
        return torch.cat(
            [
                self.ground_friction_buf,
                self.payload_mass_shift,
                self.payload_pos_local.squeeze(1),
            ],
            dim=-1,
        )

    def _encode_privileged(self, privileged_raw):
        friction = torch.clamp(privileged_raw[:, 0:1], 1.0e-2, 5.0)
        payload_mass = privileged_raw[:, 1:2]
        payload_pos = privileged_raw[:, 2:5]

        friction_linear = self._normalize_range(friction, 1.0e-2, 5.0)
        friction_log = self._normalize_range(torch.log(friction), math.log(1.0e-2), math.log(5.0))

        mass_low, mass_high = self._get_cfg_range_with_fallback(
            "payload_mass_encode_range",
            "payload_mass_range",
        )
        pos_x_low, pos_x_high = self._get_cfg_range_with_fallback(
            "payload_pos_x_encode_range",
            "payload_pos_x_range",
        )
        pos_y_low, pos_y_high = self._get_cfg_range_with_fallback(
            "payload_pos_y_encode_range",
            "payload_pos_y_range",
        )
        pos_z_low, pos_z_high = self._get_cfg_range_with_fallback(
            "payload_pos_z_encode_range",
            "payload_pos_z_range",
        )

        payload_mass_encoded = self._normalize_range(payload_mass, mass_low, mass_high)
        payload_pos_encoded = torch.cat(
            [
                self._normalize_range(payload_pos[:, 0:1], pos_x_low, pos_x_high),
                self._normalize_range(payload_pos[:, 1:2], pos_y_low, pos_y_high),
                self._normalize_range(payload_pos[:, 2:3], pos_z_low, pos_z_high),
            ],
            dim=-1,
        )

        return torch.cat(
            [
                friction_linear,
                friction_log,
                payload_mass_encoded,
                payload_pos_encoded,
            ],
            dim=-1,
        )

    def _sample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"],
            (len(envs_idx),),
            self.device,
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"],
            (len(envs_idx),),
            self.device,
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"],
            (len(envs_idx),),
            self.device,
        )
        self.commands[envs_idx, 3] = gs_rand_float(
            *self.command_cfg["height_range"],
            (len(envs_idx),),
            self.device,
        )

        self.commands[envs_idx, 4] = 0.0

        height_low = float(self.command_cfg["height_range"][0])
        height_high = float(self.command_cfg["height_range"][1])
        base_height = float(self.reward_cfg["base_height_target"])

        if abs(height_high - height_low) < 1.0e-6:
            height_diff_scale = torch.ones(
                (len(envs_idx),),
                device=self.device,
                dtype=gs.tc_float,
            )
        else:
            denom = max(
                abs(height_high - base_height),
                abs(height_low - base_height),
                1.0e-6,
            )

            height_diff_scale = (
                    0.5
                    + torch.abs(self.commands[envs_idx, 3] - base_height)
                    / denom
                    * 0.5
            )

        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale

    def _sample_jump_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.commands[envs_idx, 4] = gs_rand_float(
            *self.command_cfg["jump_range"],
            (len(envs_idx),),
            self.device,
        )

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self._update_state()
        self._compute_observations()

        return self.obs_buf, self.privileged_obs_buf

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0

        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        self.robot.set_pos(
            self.base_pos[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.robot.set_quat(
            self.base_quat[envs_idx],
            zero_velocity=False,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0

        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0

        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.jump_toggled_buf[envs_idx] = 0.0
        self.jump_target_height[envs_idx] = 0.0

        self._sample_ground_friction(envs_idx)
        self._sample_payload(envs_idx)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)

        self.commands[envs_idx, 3] = self.reward_cfg["base_height_target"]

    def step(self, actions, is_train=True):
        self.actions = torch.clip(
            actions,
            -self.env_cfg["clip_actions"],
            self.env_cfg["clip_actions"],
        )

        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"]
            + self.default_dof_pos
        )

        self.robot.control_dofs_position(
            target_dof_pos,
            self.motor_dofs,
        )

        self.scene.step()

        self.episode_length_buf += 1

        self._update_state()

        envs_idx = (
            (
                self.episode_length_buf
                % int(self.env_cfg["resampling_time_s"] / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )

        if is_train:
            self._sample_commands(envs_idx)

            random_idxs_1 = torch.randperm(self.num_envs, device=self.device)[
                : int(self.num_envs * 0.05)
            ]
            self._sample_commands(random_idxs_1)

        jump_cmd_now = (self.commands[:, 4] > 0.0).float()

        toggle_mask = (
            (self.jump_toggled_buf == 0.0)
            & (jump_cmd_now > 0.0)
        ).float()

        self.jump_toggled_buf += toggle_mask * self.reward_cfg["jump_reward_steps"]
        self.jump_toggled_buf = torch.clamp(
            self.jump_toggled_buf - 1.0,
            min=0.0,
        )

        self.jump_target_height = torch.where(
            jump_cmd_now > 0.0,
            self.commands[:, 4],
            self.jump_target_height,
        )

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )

        time_out_idx = (
            self.episode_length_buf > self.max_episode_length
        ).nonzero(as_tuple=False).flatten()

        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(
            self.reset_buf.nonzero(as_tuple=False).flatten()
        )

        self.rew_buf[:] = 0.0

        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self._compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.commands[:, 4] = 0.0

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _update_state(self):
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()

        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            )
        )

        inv_base_quat = inv_quat(self.base_quat)

        self.base_lin_vel[:] = transform_by_quat(
            self.robot.get_vel(),
            inv_base_quat,
        )
        self.base_ang_vel[:] = transform_by_quat(
            self.robot.get_ang(),
            inv_base_quat,
        )
        self.projected_gravity = transform_by_quat(
            self.global_gravity,
            inv_base_quat,
        )

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

    def _compute_observations(self):
        action_obs = (
            self.last_actions
            if self.env_cfg.get("use_last_actions_in_obs", True)
            else self.actions
        )

        base_obs = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],            # 3
                self.projected_gravity,                                    # 3
                self.commands * self.commands_scale,                       # 5
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                 # 12
                action_obs,                                                # 12
                (
                    self.jump_toggled_buf
                    / self.reward_cfg["jump_reward_steps"]
                ).unsqueeze(-1),                                           # 1
            ],
            dim=-1,
        )

        privileged_raw = self._get_privileged_raw()
        privileged_encoded = self._encode_privileged(privileged_raw)

        self.obs_buf = torch.cat(
            [
                base_obs,
                privileged_encoded,
            ],
            dim=-1,
        )

        self.privileged_obs_buf = torch.cat(
            [
                self.obs_buf,
                privileged_raw,
            ],
            dim=-1,
        )

        self.obs_buf = torch.clip(
            self.obs_buf,
            -self.obs_cfg.get("clip_observations", 100.0),
            self.obs_cfg.get("clip_observations", 100.0),
        )
        self.privileged_obs_buf = torch.clip(
            self.privileged_obs_buf,
            -self.obs_cfg.get("clip_observations", 100.0),
            self.obs_cfg.get("clip_observations", 100.0),
        )


    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2]
        )
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(
            torch.square(self.last_actions - self.actions),
            dim=1,
        )

    def _reward_similar_to_default(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.sum(
            torch.abs(self.dof_pos - self.default_dof_pos),
            dim=1,
        )

    def _reward_base_height(self):
        active_mask = (self.jump_toggled_buf < 0.01).float()
        return active_mask * torch.square(
            self.base_pos[:, 2] - self.commands[:, 3]
        )

    def _reward_jump_height_tracking(self):
        mask = (
            (self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"])
            & (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"])
        )
        target_height = self.jump_target_height

        height_diff = torch.exp(
            -torch.square(self.base_pos[:, 2] - target_height)
        )

        return mask.float() * height_diff

    def _reward_jump_height_achievement(self):
        mask = (
            (self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"])
            & (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"])
        )
        target_height = self.jump_target_height

        binary_bonus = (
            torch.abs(self.base_pos[:, 2] - target_height) < 0.2
        ).float()

        return mask.float() * binary_bonus

    def _reward_jump_speed(self):
        mask = (
            (self.jump_toggled_buf >= 0.3 * self.reward_cfg["jump_reward_steps"])
            & (self.jump_toggled_buf < 0.6 * self.reward_cfg["jump_reward_steps"])
        )

        return mask.float() * torch.exp(self.base_lin_vel[:, 2]) * 0.2

    def _reward_jump_landing(self):
        mask = (
            self.jump_toggled_buf
            >= 0.6 * self.reward_cfg["jump_reward_steps"]
        )

        height_error = -torch.square(
            self.base_pos[:, 2] - self.reward_cfg["base_height_target"]
        )

        return mask.float() * height_error