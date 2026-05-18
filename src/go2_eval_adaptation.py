import argparse
import math
import os
import pickle
import time

import torch
import torch.nn as nn
import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env


class AdaptationMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def gs_backend(device):
    if device == "cuda:0":
        return gs.constants.backend.gpu
    return gs.constants.backend.cpu


def get_cfg_range(env_cfg, key, default=(0.0, 0.0)):
    values = env_cfg.get(key, default)

    if len(values) != 2:
        raise ValueError(f"{key} must have exactly two values, got {values}")

    return float(values[0]), float(values[1])


def unnormalize_range(values, low, high):
    return (values + 1.0) * 0.5 * (high - low) + low


def decode_fixed_payload_z(z, env_cfg):
    if z.shape[-1] != 3:
        raise ValueError(f"Expected z dim 3, got {z.shape[-1]}")

    z = torch.clamp(z, -1.0, 1.0)

    friction_low, friction_high = get_cfg_range(
        env_cfg,
        "ground_friction_encode_range",
        get_cfg_range(env_cfg, "ground_friction_range", [0.1, 1.0]),
    )

    friction_linear = unnormalize_range(
        z[:, 0:1],
        friction_low,
        friction_high,
    )

    friction_log_value = unnormalize_range(
        z[:, 1:2],
        math.log(friction_low),
        math.log(friction_high),
    )
    friction_log = torch.exp(friction_log_value)

    mass_low, mass_high = get_cfg_range(
        env_cfg,
        "payload_mass_encode_range",
        get_cfg_range(env_cfg, "payload_mass_range"),
    )

    payload_mass = unnormalize_range(
        z[:, 2:3],
        mass_low,
        mass_high,
    )

    return torch.cat(
        [
            friction_linear,
            friction_log,
            payload_mass,
        ],
        dim=-1,
    )


def apply_fixed_command(env, vx, vy, yaw_rate, height):
    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 2] = yaw_rate
    env.commands[:, 3] = height

    if env.commands.shape[1] > 4:
        env.commands[:, 4] = 0.0

    env._compute_observations()

    return env.obs_buf, env.privileged_obs_buf


def make_policy_obs(base_obs, z_policy):
    return torch.cat([base_obs, z_policy], dim=-1)


def load_adaptation_model(path, device):
    checkpoint = torch.load(path, map_location=device)

    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])

    model = AdaptationMLP(
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    window_len = int(checkpoint["window_len"])
    action_key = checkpoint.get("action_key", "executed_actions")

    return model, checkpoint, window_len, action_key


def check_fixed_payload_layout(obs_cfg, adaptation_ckpt=None):
    base_num_obs = int(obs_cfg["base_num_obs"])
    privileged_encoder_dim = int(obs_cfg["privileged_encoder_dim"])
    privileged_raw_dim = int(obs_cfg["privileged_raw_dim"])
    num_obs = int(obs_cfg["num_obs"])
    num_privileged_obs = int(obs_cfg["num_privileged_obs"])

    expected_num_obs = base_num_obs + privileged_encoder_dim
    expected_num_privileged_obs = expected_num_obs + privileged_raw_dim

    if privileged_encoder_dim != 3:
        raise RuntimeError(
            f"Bad privileged_encoder_dim={privileged_encoder_dim}. "
            "This eval expects fixed payload-position layout with encoded dim 3."
        )

    if privileged_raw_dim != 2:
        raise RuntimeError(
            f"Bad privileged_raw_dim={privileged_raw_dim}. "
            "This eval expects fixed payload-position layout with raw dim 2."
        )

    if num_obs != expected_num_obs:
        raise RuntimeError(
            f"Bad num_obs={num_obs}, expected {expected_num_obs}."
        )

    if num_privileged_obs != expected_num_privileged_obs:
        raise RuntimeError(
            f"Bad num_privileged_obs={num_privileged_obs}, "
            f"expected {expected_num_privileged_obs}."
        )

    if adaptation_ckpt is not None:
        output_dim = int(adaptation_ckpt["output_dim"])

        if output_dim != privileged_encoder_dim:
            raise RuntimeError(
                f"Bad adapter output_dim={output_dim}, "
                f"expected {privileged_encoder_dim}."
            )

    return {
        "base_num_obs": base_num_obs,
        "privileged_encoder_dim": privileged_encoder_dim,
        "privileged_raw_dim": privileged_raw_dim,
        "num_obs": num_obs,
        "num_privileged_obs": num_privileged_obs,
    }


def make_live_plot(enabled=True):
    if not enabled:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] disabled: {e}")
        return None

    plt.ion()

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    ax_y = axes[0]
    ax_vx = axes[1]
    ax_fric_linear = axes[2]
    ax_fric_log = axes[3]
    ax_mass = axes[4]

    line_y, = ax_y.plot([], [], label="drift_y")
    ax_y.axhline(0.0, linestyle="--", linewidth=1.0, label="y = 0")
    ax_y.set_ylabel("y drift [m]")
    ax_y.grid(True)
    ax_y.legend(loc="upper right")

    line_vx_world, = ax_vx.plot([], [], label="vx_world")
    line_vx_body, = ax_vx.plot([], [], label="vx_body")
    line_vx_cmd, = ax_vx.plot([], [], linestyle="--", label="vx zadane")
    ax_vx.set_ylabel("vx [m/s]")
    ax_vx.grid(True)
    ax_vx.legend(loc="upper right")

    line_fric_linear, = ax_fric_linear.plot([], [], label="RMA friction from linear z")
    line_fric_true_1, = ax_fric_linear.plot([], [], linestyle="--", label="true friction")
    ax_fric_linear.set_ylabel("friction")
    ax_fric_linear.grid(True)
    ax_fric_linear.legend(loc="upper right")

    line_fric_log, = ax_fric_log.plot([], [], label="RMA friction from log z")
    line_fric_true_2, = ax_fric_log.plot([], [], linestyle="--", label="true friction")
    ax_fric_log.set_ylabel("friction")
    ax_fric_log.grid(True)
    ax_fric_log.legend(loc="upper right")

    line_mass, = ax_mass.plot([], [], label="RMA payload mass")
    line_mass_true, = ax_mass.plot([], [], linestyle="--", label="true payload mass")
    ax_mass.set_ylabel("mass [kg]")
    ax_mass.set_xlabel("time [s]")
    ax_mass.grid(True)
    ax_mass.legend(loc="upper right")

    fig.suptitle("Go2 RMA transition processes")
    fig.tight_layout()

    plt.show(block=False)

    return {
        "plt": plt,
        "fig": fig,
        "axes": axes,
        "t": [],
        "drift_y": [],
        "vx_world": [],
        "vx_body": [],
        "vx_cmd": [],
        "friction_linear": [],
        "friction_log": [],
        "friction_true": [],
        "payload_mass": [],
        "payload_mass_true": [],
        "lines": {
            "drift_y": line_y,
            "vx_world": line_vx_world,
            "vx_body": line_vx_body,
            "vx_cmd": line_vx_cmd,
            "friction_linear": line_fric_linear,
            "friction_true_1": line_fric_true_1,
            "friction_log": line_fric_log,
            "friction_true_2": line_fric_true_2,
            "payload_mass": line_mass,
            "payload_mass_true": line_mass_true,
        },
    }


def update_live_plot(
    plot_state,
    t,
    drift_y,
    vx_world,
    vx_body,
    vx_cmd,
    raw_true,
    z_policy_decoded,
    falls,
    mode,
):
    if plot_state is None:
        return

    raw_true_np = raw_true.detach().cpu().float().numpy()
    decoded_np = z_policy_decoded.detach().cpu().float().numpy()

    true_friction = float(raw_true_np[0])
    true_payload_mass = float(raw_true_np[1])

    friction_linear = float(decoded_np[0])
    friction_log = float(decoded_np[1])
    payload_mass = float(decoded_np[2])

    plot_state["t"].append(float(t))
    plot_state["drift_y"].append(float(drift_y))
    plot_state["vx_world"].append(float(vx_world))
    plot_state["vx_body"].append(float(vx_body))
    plot_state["vx_cmd"].append(float(vx_cmd))
    plot_state["friction_linear"].append(friction_linear)
    plot_state["friction_log"].append(friction_log)
    plot_state["friction_true"].append(true_friction)
    plot_state["payload_mass"].append(payload_mass)
    plot_state["payload_mass_true"].append(true_payload_mass)

    t_values = plot_state["t"]
    lines = plot_state["lines"]

    lines["drift_y"].set_data(t_values, plot_state["drift_y"])

    lines["vx_world"].set_data(t_values, plot_state["vx_world"])
    lines["vx_body"].set_data(t_values, plot_state["vx_body"])
    lines["vx_cmd"].set_data(t_values, plot_state["vx_cmd"])

    lines["friction_linear"].set_data(t_values, plot_state["friction_linear"])
    lines["friction_true_1"].set_data(t_values, plot_state["friction_true"])

    lines["friction_log"].set_data(t_values, plot_state["friction_log"])
    lines["friction_true_2"].set_data(t_values, plot_state["friction_true"])

    lines["payload_mass"].set_data(t_values, plot_state["payload_mass"])
    lines["payload_mass_true"].set_data(t_values, plot_state["payload_mass_true"])

    for ax in plot_state["axes"]:
        ax.relim()
        ax.autoscale_view()

    if len(t_values) > 2:
        plot_state["axes"][-1].set_xlim(t_values[0], t_values[-1])

    plot_state["fig"].suptitle(
        f"Go2 RMA transition processes | mode={mode} | falls={falls}"
    )
    plot_state["fig"].canvas.draw()
    plot_state["fig"].canvas.flush_events()
    plot_state["plt"].pause(0.001)


def save_final_plot(plot_state, path):
    if plot_state is None or path is None:
        return

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plot_state["fig"].savefig(path, dpi=160)
    print(f"[plot] saved to: {path}")


def read_step_metrics(env, infos, done):
    if done:
        vx_world = float(infos["terminal_vx_world"][0].item())
        vy_world = float(infos["terminal_vy_world"][0].item())
        vx_body = float(infos["terminal_vx_body"][0].item())
        vy_body = float(infos["terminal_vy_body"][0].item())
        z = float(infos["terminal_z"][0].item())
        lateral_error = float(infos["terminal_lateral_error"][0].item())
        yaw_accum = float(infos["terminal_yaw_accum"][0].item())
        roll = float(infos["terminal_roll"][0].item())
        pitch = float(infos["terminal_pitch"][0].item())
    else:
        vx_world = float(env.base_lin_vel_world[0, 0].item())
        vy_world = float(env.base_lin_vel_world[0, 1].item())
        vx_body = float(env.base_lin_vel[0, 0].item())
        vy_body = float(env.base_lin_vel[0, 1].item())
        z = float(env.base_pos[0, 2].item())
        lateral_error = float(
            (env.base_pos[0, 1] - env.episode_start_y[0]).item()
        )
        yaw_accum = float(env.yaw_angle_accum[0].item())
        roll = float(env.base_euler[0, 0].item())
        pitch = float(env.base_euler[0, 1].item())

    return {
        "vx_world": vx_world,
        "vy_world": vy_world,
        "vx_body": vx_body,
        "vy_body": vy_body,
        "z": z,
        "lateral_error": lateral_error,
        "yaw_accum": yaw_accum,
        "roll": roll,
        "pitch": pitch,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)

    parser.add_argument(
        "--mode",
        type=str,
        default="rma",
        choices=["teacher", "rma", "zero"],
        help=(
            "teacher = use true privileged z, "
            "rma = use adapter z_hat, "
            "zero = use z_hat = zeros"
        ),
    )

    parser.add_argument(
        "--adaptation_path",
        type=str,
        default=None,
        help="Required only for --mode rma",
    )

    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])

    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--payload_mass", type=float, default=1.0)

    parser.add_argument("--payload_pos_x", type=float, default=0.0)
    parser.add_argument("--payload_pos_y", type=float, default=0.0)
    parser.add_argument("--payload_pos_z", type=float, default=0.05)

    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw_rate", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=0.3)

    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--warmup_steps", type=int, default=100)

    parser.add_argument(
        "--bootstrap_true_steps",
        type=int,
        default=0,
        help="For --mode rma: use true z for the first N steps of every episode.",
    )

    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--plot_every", type=int, default=5)
    parser.add_argument("--print_every", type=int, default=250)
    parser.add_argument("--sleep", type=float, default=0.016)

    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="Optional path to save final transition plot, e.g. results/rma_transition.png",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "rma" and args.adaptation_path is None:
        raise ValueError("--adaptation_path is required for --mode rma")

    if args.mode != "rma" and args.bootstrap_true_steps > 0:
        print("[warning] --bootstrap_true_steps is only used in --mode rma")

    gs.init(
        logging_level="warning",
        backend=gs_backend(args.device),
    )

    device = torch.device(args.device)

    log_dir = os.path.join("logs", args.exp_name)
    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    adaptation_model = None
    adaptation_ckpt = None
    window_len = 1
    action_key = "executed_actions"

    if args.mode == "rma":
        if not os.path.exists(args.adaptation_path):
            raise FileNotFoundError(
                f"Adaptation checkpoint not found: {args.adaptation_path}"
            )

        adaptation_model, adaptation_ckpt, window_len, action_key = load_adaptation_model(
            args.adaptation_path,
            device,
        )

    dims = check_fixed_payload_layout(
        obs_cfg=obs_cfg,
        adaptation_ckpt=adaptation_ckpt,
    )

    base_num_obs = dims["base_num_obs"]
    privileged_encoder_dim = dims["privileged_encoder_dim"]
    privileged_raw_dim = dims["privileged_raw_dim"]
    num_obs = dims["num_obs"]
    num_privileged_obs = dims["num_privileged_obs"]

    num_actions = int(env_cfg["num_actions"])

    if args.mode == "rma":
        expected_input_dim = window_len * (base_num_obs + num_actions)
        actual_input_dim = int(adaptation_ckpt["input_dim"])

        if actual_input_dim != expected_input_dim:
            raise RuntimeError(
                f"Bad adaptation input_dim: checkpoint={actual_input_dim}, "
                f"expected={expected_input_dim}"
            )

        if action_key != "executed_actions":
            print(
                "[warning] Adapter was trained with action_key="
                f"{action_key!r}. Online eval uses previous executed actions."
            )

    local_env_cfg = dict(env_cfg)
    local_reward_cfg = dict(reward_cfg)

    local_reward_cfg["reward_scales"] = {}

    local_env_cfg["ground_friction"] = float(args.friction)
    local_env_cfg["payload_mass_range"] = [
        float(args.payload_mass),
        float(args.payload_mass),
    ]
    local_env_cfg["payload_pos"] = [
        float(args.payload_pos_x),
        float(args.payload_pos_y),
        float(args.payload_pos_z),
    ]

    print("============================================================")
    print("Go2 adaptation eval")
    print(f"Teacher experiment: {args.exp_name}")
    print(f"Teacher checkpoint: {ckpt_path}")
    print(f"Mode: {args.mode}")

    if args.mode == "rma":
        print(f"Adaptation checkpoint: {args.adaptation_path}")
        print(f"Adapter window_len: {window_len}")
        print(f"Adapter action_key: {action_key}")
        print(f"Bootstrap true steps: {args.bootstrap_true_steps}")
    elif args.mode == "teacher":
        print("Policy z: true privileged z")
    else:
        print("Policy z: zeros")

    print()
    print(f"friction={args.friction}")
    print(f"payload_mass={args.payload_mass}")
    print(
        "payload_pos="
        f"({args.payload_pos_x}, {args.payload_pos_y}, {args.payload_pos_z})"
    )
    print(
        "command: "
        f"vx={args.vx}, vy={args.vy}, yaw_rate={args.yaw_rate}, height={args.height}"
    )
    print()
    print(f"base_num_obs={base_num_obs}")
    print(f"privileged_encoder_dim={privileged_encoder_dim}")
    print(f"privileged_raw_dim={privileged_raw_dim}")
    print(f"num_obs={num_obs}")
    print(f"num_privileged_obs={num_privileged_obs}")
    print("============================================================")

    env = Go2Env(
        num_envs=1,
        env_cfg=local_env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=local_reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.no_render,
        device=args.device,
    )

    runner = OnPolicyRunner(
        env,
        train_cfg,
        log_dir,
        device=args.device,
    )

    runner.load(
        ckpt_path,
        load_optimizer=False,
    )

    policy = runner.get_inference_policy(device=args.device)

    history_base_obs = torch.zeros(
        (1, window_len, base_num_obs),
        device=device,
        dtype=torch.float32,
    )

    history_actions = torch.zeros(
        (1, window_len, num_actions),
        device=device,
        dtype=torch.float32,
    )

    plot_state = make_live_plot(
        enabled=not args.no_plot,
    )

    obs, privileged_obs = env.reset()
    obs, privileged_obs = apply_fixed_command(
        env,
        args.vx,
        args.vy,
        args.yaw_rate,
        args.height,
    )

    vx_world_sum = 0.0
    vy_world_sum = 0.0
    vx_body_sum = 0.0
    vy_body_sum = 0.0
    z_sum = 0.0
    action_sum = 0.0

    distance_x_integral = 0.0
    distance_y_integral = 0.0

    max_abs_y = 0.0
    max_abs_yaw = 0.0
    max_abs_roll = 0.0
    max_abs_pitch = 0.0

    z_mse_sum = 0.0
    z_mae_sum = 0.0
    z_count = 0

    per_dim_abs_sum = torch.zeros(
        (privileged_encoder_dim,),
        dtype=torch.float64,
    )

    n = 0
    falls = 0
    reset_roll_count = 0
    reset_pitch_count = 0
    reset_timeout_count = 0

    current_episode_len = 0
    episode_len_sum = 0
    completed_episodes = 0

    episode_step = 0

    target_names = [
        "friction_linear",
        "friction_log",
        "payload_mass_encoded",
    ]

    last_z_true = None
    last_z_policy = None
    last_raw_true = None
    last_z_policy_decoded = None

    with torch.no_grad():
        for step_i in range(args.steps):
            base_obs = obs[:, :base_num_obs]
            z_true = obs[:, base_num_obs:base_num_obs + privileged_encoder_dim]
            raw_true = privileged_obs[:, num_obs:num_obs + privileged_raw_dim]

            previous_executed_action = env.last_actions.clone()

            history_base_obs = torch.roll(
                history_base_obs,
                shifts=-1,
                dims=1,
            )
            history_actions = torch.roll(
                history_actions,
                shifts=-1,
                dims=1,
            )

            history_base_obs[:, -1, :] = base_obs
            history_actions[:, -1, :] = previous_executed_action

            if args.mode == "teacher":
                z_policy = z_true
            elif args.mode == "zero":
                z_policy = torch.zeros_like(z_true)
            else:
                adaptation_input = torch.cat(
                    [
                        history_base_obs,
                        history_actions,
                    ],
                    dim=-1,
                ).reshape(1, -1)

                z_pred = adaptation_model(adaptation_input)
                z_pred = torch.clamp(z_pred, -1.0, 1.0)

                if episode_step < args.bootstrap_true_steps:
                    z_policy = z_true
                else:
                    z_policy = z_pred

            obs_for_policy = make_policy_obs(
                base_obs=base_obs,
                z_policy=z_policy,
            )

            actions = policy(obs_for_policy)

            if step_i >= args.warmup_steps:
                diff = z_policy - z_true

                z_mse_sum += torch.sum(diff ** 2).item()
                z_mae_sum += torch.sum(torch.abs(diff)).item()
                per_dim_abs_sum += torch.abs(diff[0]).detach().cpu().double()
                z_count += privileged_encoder_dim

            obs, privileged_obs, rewards, dones, infos = env.step(
                actions,
                is_train=False,
            )

            done = bool(dones[0].item())

            step_metrics = read_step_metrics(
                env=env,
                infos=infos,
                done=done,
            )

            current_episode_len += 1

            z_policy_decoded = decode_fixed_payload_z(
                z_policy,
                local_env_cfg,
            )

            if step_i >= args.warmup_steps:
                vx_world_sum += step_metrics["vx_world"]
                vy_world_sum += step_metrics["vy_world"]
                vx_body_sum += step_metrics["vx_body"]
                vy_body_sum += step_metrics["vy_body"]
                z_sum += step_metrics["z"]

                distance_x_integral += step_metrics["vx_world"] * env.dt
                distance_y_integral += step_metrics["vy_world"] * env.dt

                action_sum += float(torch.mean(torch.abs(actions[0])).item())

                max_abs_y = max(max_abs_y, abs(step_metrics["lateral_error"]))
                max_abs_yaw = max(max_abs_yaw, abs(step_metrics["yaw_accum"]))
                max_abs_roll = max(max_abs_roll, abs(step_metrics["roll"]))
                max_abs_pitch = max(max_abs_pitch, abs(step_metrics["pitch"]))

                n += 1

            last_z_true = z_true.detach().clone()
            last_z_policy = z_policy.detach().clone()
            last_raw_true = raw_true.detach().clone()
            last_z_policy_decoded = z_policy_decoded.detach().clone()

            current_time = step_i * env.dt

            if step_i % args.plot_every == 0:
                update_live_plot(
                    plot_state=plot_state,
                    t=current_time,
                    drift_y=step_metrics["lateral_error"],
                    vx_world=step_metrics["vx_world"],
                    vx_body=step_metrics["vx_body"],
                    vx_cmd=args.vx,
                    raw_true=raw_true[0],
                    z_policy_decoded=z_policy_decoded[0],
                    falls=falls,
                    mode=args.mode,
                )

            if args.print_every > 0 and step_i > 0 and step_i % args.print_every == 0:
                avg_vx_world_live = vx_world_sum / max(n, 1)
                avg_vx_body_live = vx_body_sum / max(n, 1)
                avg_vy_world_live = vy_world_sum / max(n, 1)

                print(
                    f"step={step_i:5d} | "
                    f"ep_step={episode_step:4d} | "
                    f"vx_w={avg_vx_world_live:.3f} | "
                    f"vx_b={avg_vx_body_live:.3f} | "
                    f"vy_w={avg_vy_world_live:.3f} | "
                    f"max_y={max_abs_y:.3f} | "
                    f"falls={falls}"
                )

            if done:
                falls += 1
                completed_episodes += 1
                episode_len_sum += current_episode_len

                if bool(infos["reset_roll"][0].item()):
                    reset_roll_count += 1
                if bool(infos["reset_pitch"][0].item()):
                    reset_pitch_count += 1
                if bool(infos["reset_timeout"][0].item()):
                    reset_timeout_count += 1

                current_episode_len = 0
                episode_step = 0

                history_base_obs[:] = 0.0
                history_actions[:] = 0.0
            else:
                episode_step += 1

            obs, privileged_obs = apply_fixed_command(
                env,
                args.vx,
                args.vy,
                args.yaw_rate,
                args.height,
            )

            if not args.no_render and args.sleep > 0.0:
                time.sleep(args.sleep)

    if current_episode_len > 0:
        completed_episodes += 1
        episode_len_sum += current_episode_len

    avg_vx_world = vx_world_sum / max(n, 1)
    avg_vy_world = vy_world_sum / max(n, 1)
    avg_vx_body = vx_body_sum / max(n, 1)
    avg_vy_body = vy_body_sum / max(n, 1)
    avg_z = z_sum / max(n, 1)
    avg_action = action_sum / max(n, 1)

    avg_episode_len = episode_len_sum / max(completed_episodes, 1)

    z_mse = z_mse_sum / max(z_count, 1)
    z_mae = z_mae_sum / max(z_count, 1)
    per_dim_mae = per_dim_abs_sum / max(n, 1)

    per_dim_text = ", ".join(
        f"{name}={value:.6f}"
        for name, value in zip(target_names, per_dim_mae.tolist())
    )

    print()
    print("============================================================")
    print("Evaluation result")
    print("============================================================")
    print(f"mode:                 {args.mode}")
    print(f"friction:             {args.friction:.4f}")
    print(f"payload_mass:         {args.payload_mass:.4f}")
    print(f"command_vx:           {args.vx:.4f}")
    print()
    print(f"avg_vx_world:         {avg_vx_world:.4f}")
    print(f"avg_vy_world:         {avg_vy_world:.4f}")
    print(f"avg_vx_body:          {avg_vx_body:.4f}")
    print(f"avg_vy_body:          {avg_vy_body:.4f}")
    print(f"avg_z:                {avg_z:.4f}")
    print(f"avg_action_abs:       {avg_action:.4f}")
    print()
    print(f"distance_x_integral:  {distance_x_integral:.4f}")
    print(f"distance_y_integral:  {distance_y_integral:.4f}")
    print(f"max_abs_y:            {max_abs_y:.4f}")
    print(f"max_abs_yaw:          {max_abs_yaw:.4f}")
    print(f"max_abs_roll:         {max_abs_roll:.4f}")
    print(f"max_abs_pitch:        {max_abs_pitch:.4f}")
    print()
    print(f"falls:                {falls}")
    print(f"reset_roll_count:     {reset_roll_count}")
    print(f"reset_pitch_count:    {reset_pitch_count}")
    print(f"reset_timeout_count:  {reset_timeout_count}")
    print(f"completed_episodes:   {completed_episodes}")
    print(f"avg_episode_len:      {avg_episode_len:.1f}")
    print()
    print(f"z_mse:                {z_mse:.6f}")
    print(f"z_mae:                {z_mae:.6f}")
    print(f"z_per_dim_mae:        [{per_dim_text}]")

    if last_z_true is not None and last_z_policy is not None:
        print()
        print("Last z values")
        print(f"z_true:               {last_z_true[0].detach().cpu().float().numpy()}")
        print(f"z_policy:             {last_z_policy[0].detach().cpu().float().numpy()}")
        print(f"raw_true:             {last_raw_true[0].detach().cpu().float().numpy()}")
        print(
            "z_policy_decoded:    "
            f"{last_z_policy_decoded[0].detach().cpu().float().numpy()}"
        )

    print("============================================================")

    save_final_plot(plot_state, args.save_plot)

    if plot_state is not None:
        print("Close matplotlib window to finish.")
        plot_state["plt"].ioff()
        plot_state["plt"].show()


if __name__ == "__main__":
    main()