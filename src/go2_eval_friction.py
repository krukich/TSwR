import argparse
import os
import pickle
import time

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env


def parse_float_list(text: str):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        values.append(value)
    return values


def parse_friction_list(text: str):
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        value = max(value, 1.0e-2)
        value = min(value, 5.0)
        values.append(value)
    return values


def run_one_friction(
    friction,
    payload_mass,
    payload_pos_x,
    payload_pos_y,
    payload_pos_z,
    env_cfg,
    obs_cfg,
    reward_cfg,
    command_cfg,
    train_cfg,
    log_dir,
    ckpt,
    device,
    vx,
    vy,
    yaw_rate,
    height,
    steps,
    warmup_steps,
    render,
    sleep,
):
    local_env_cfg = dict(env_cfg)
    local_reward_cfg = dict(reward_cfg)

    local_reward_cfg["reward_scales"] = {}

    local_env_cfg["termination_if_roll_greater_than"] = 50
    local_env_cfg["termination_if_pitch_greater_than"] = 50

    local_env_cfg["ground_friction"] = float(friction)
    local_env_cfg["payload_mass_range"] = [float(payload_mass), float(payload_mass)]
    local_env_cfg["payload_pos_x_range"] = [float(payload_pos_x), float(payload_pos_x)]
    local_env_cfg["payload_pos_y_range"] = [float(payload_pos_y), float(payload_pos_y)]
    local_env_cfg["payload_pos_z_range"] = [float(payload_pos_z), float(payload_pos_z)]

    env = Go2Env(
        num_envs=1,
        env_cfg=local_env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=local_reward_cfg,
        command_cfg=command_cfg,
        show_viewer=render,
        device=device,
    )

    runner = OnPolicyRunner(
        env,
        train_cfg,
        log_dir,
        device=device,
    )

    ckpt_path = os.path.join(log_dir, f"model_{ckpt}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()

    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 2] = yaw_rate
    env.commands[:, 3] = height
    env.commands[:, 4] = 0.0
    env._compute_observations()
    obs = env.obs_buf

    x_start = float(env.base_pos[0, 0].item())
    y_start = float(env.base_pos[0, 1].item())

    vx_sum = 0.0
    vy_sum = 0.0
    z_sum = 0.0
    action_sum = 0.0
    n = 0

    falls = 0
    episode_len_sum = 0
    current_ep_len = 0

    max_abs_y = 0.0
    final_x = x_start
    final_y = y_start

    with torch.no_grad():
        for step_i in range(steps):
            actions = policy(obs)

            obs, _, rewards, dones, infos = env.step(actions, is_train=False)

            env.commands[:, 0] = vx
            env.commands[:, 1] = vy
            env.commands[:, 2] = yaw_rate
            env.commands[:, 3] = height
            env.commands[:, 4] = 0.0
            env._compute_observations()
            obs = env.obs_buf

            current_ep_len += 1

            x = float(env.base_pos[0, 0].item())
            y = float(env.base_pos[0, 1].item())
            z = float(env.base_pos[0, 2].item())

            final_x = x
            final_y = y
            max_abs_y = max(max_abs_y, abs(y - y_start))

            if step_i >= warmup_steps:
                vx_sum += float(env.base_lin_vel[0, 0].item())
                vy_sum += float(env.base_lin_vel[0, 1].item())
                z_sum += z
                action_sum += float(torch.mean(torch.abs(actions[0])).item())
                n += 1

            if bool(dones[0].item()):
                falls += 1
                episode_len_sum += current_ep_len
                current_ep_len = 0

                # Новый старт после reset.
                x_start = float(env.base_pos[0, 0].item())
                y_start = float(env.base_pos[0, 1].item())
                final_x = x_start
                final_y = y_start
                max_abs_y = 0.0

            if render and sleep > 0.0:
                time.sleep(sleep)

    if current_ep_len > 0:
        episode_len_sum += current_ep_len

    avg_vx = vx_sum / max(n, 1)
    avg_vy = vy_sum / max(n, 1)
    avg_z = z_sum / max(n, 1)
    avg_action = action_sum / max(n, 1)

    distance_x = final_x - x_start
    drift_y = final_y - y_start

    avg_episode_len = episode_len_sum / max(falls + 1, 1)

    return {
        "friction": friction,
        "payload_mass": payload_mass,
        "payload_pos_x": payload_pos_x,
        "payload_pos_y": payload_pos_y,
        "payload_pos_z": payload_pos_z,
        "avg_vx": avg_vx,
        "avg_vy": avg_vy,
        "avg_z": avg_z,
        "avg_action": avg_action,
        "distance_x": distance_x,
        "drift_y": drift_y,
        "max_abs_y": max_abs_y,
        "falls": falls,
        "avg_episode_len": avg_episode_len,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])

    parser.add_argument("--frictions", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--payload_masses", type=str, default="0.0")
    parser.add_argument("--payload_pos_x", type=float, default=0.0)
    parser.add_argument("--payload_pos_y", type=float, default=0.0)
    parser.add_argument("--payload_pos_z", type=float, default=0.05)

    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw_rate", type=float, default=0.0)
    parser.add_argument("--height", type=float, default=0.3)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)

    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.016)

    args = parser.parse_args()

    backend = gs.constants.backend.gpu if args.device == "cuda:0" else gs.constants.backend.cpu
    gs.init(backend=backend, logging_level="warning")

    log_dir = os.path.join("logs", args.exp_name)
    cfg_path = os.path.join(log_dir, "cfgs.pkl")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    friction_values = parse_friction_list(args.frictions)
    payload_masses = parse_float_list(args.payload_masses)

    print("============================================================")
    print("Go2 friction eval")
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: model_{args.ckpt}.pt")
    print(f"Command: vx={args.vx}, vy={args.vy}, yaw_rate={args.yaw_rate}, height={args.height}")
    print(f"Frictions: {friction_values}")
    print(f"Payload masses: {payload_masses}")
    print(
        "Payload position: "
        f"x={args.payload_pos_x}, y={args.payload_pos_y}, z={args.payload_pos_z}"
    )
    print("============================================================")
    print(
        f"{'payload':>8s} | "
        f"{'friction':>8s} | "
        f"{'avg_vx':>8s} | "
        f"{'avg_vy':>8s} | "
        f"{'avg_z':>8s} | "
        f"{'act':>8s} | "
        f"{'dx':>8s} | "
        f"{'dy':>8s} | "
        f"{'max|y|':>8s} | "
        f"{'falls':>5s} | "
        f"{'ep_len':>8s}"
    )
    print("-" * 116)

    results = []

    for payload_mass in payload_masses:
        for friction in friction_values:
            result = run_one_friction(
                friction=friction,
                payload_mass=payload_mass,
                payload_pos_x=args.payload_pos_x,
                payload_pos_y=args.payload_pos_y,
                payload_pos_z=args.payload_pos_z,
                env_cfg=env_cfg,
                obs_cfg=obs_cfg,
                reward_cfg=reward_cfg,
                command_cfg=command_cfg,
                train_cfg=train_cfg,
                log_dir=log_dir,
                ckpt=args.ckpt,
                device=args.device,
                vx=args.vx,
                vy=args.vy,
                yaw_rate=args.yaw_rate,
                height=args.height,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
                render=not args.no_render,
                sleep=args.sleep,
            )

            results.append(result)

            print(
                f"{result['payload_mass']:8.3f} | "
                f"{result['friction']:8.3f} | "
                f"{result['avg_vx']:8.3f} | "
                f"{result['avg_vy']:8.3f} | "
                f"{result['avg_z']:8.3f} | "
                f"{result['avg_action']:8.3f} | "
                f"{result['distance_x']:8.3f} | "
                f"{result['drift_y']:8.3f} | "
                f"{result['max_abs_y']:8.3f} | "
                f"{result['falls']:5d} | "
                f"{result['avg_episode_len']:8.1f}"
            )

        print("-" * 116)

    best = min(
        results,
        key=lambda r: abs(r["avg_vx"] - args.vx) + 0.2 * r["falls"] + abs(r["drift_y"]),
    )

    print()
    print(
        "Best approximate setup by simple score: "
        f"payload={best['payload_mass']:.3f}, "
        f"friction={best['friction']:.3f} "
        f"(avg_vx={best['avg_vx']:.3f}, falls={best['falls']}, drift_y={best['drift_y']:.3f})"
    )


if __name__ == "__main__":
    main()
