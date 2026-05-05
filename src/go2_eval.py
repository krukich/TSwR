import argparse
import os
import pickle

import genesis as gs
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])

    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--height", type=float, default=0.3)

    args = parser.parse_args()

    backend = (
        gs.constants.backend.gpu
        if args.device == "cuda:0"
        else gs.constants.backend.cpu
    )
    gs.init(backend=backend)

    log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Same as Argo eval: no rewards, very soft termination.
    reward_cfg["reward_scales"] = {}
    env_cfg["termination_if_roll_greater_than"] = 50
    env_cfg["termination_if_pitch_greater_than"] = 50

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device=args.device,
    )

    runner = OnPolicyRunner(
        env,
        train_cfg,
        log_dir,
        device=args.device,
    )

    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=args.device)

    obs, _ = env.reset()

    iteration = 0
    lin_x_range = [0.5, 4.0]

    with torch.no_grad():
        while True:
            actions = policy(obs)

            if args.vx is None:
                lin_x = (
                    lin_x_range[0]
                    + (lin_x_range[1] - lin_x_range[0])
                    * (np.sin(2 * np.pi * iteration / 600) + 1)
                    / 2
                )
            else:
                lin_x = args.vx

            lin_x = float(lin_x)

            env.commands = torch.tensor(
                [[lin_x, 0.0, 0.0, args.height, 0.0]],
                device=args.device,
                dtype=torch.float32,
            )

            obs, _, rews, dones, infos = env.step(actions, is_train=False)

            if iteration % 50 == 0:
                action_abs = torch.abs(actions[0]).detach().cpu()

                print(
                    f"iter={iteration} "
                    f"cmd_vx={lin_x:.3f} "
                    f"vx={float(env.base_lin_vel[0, 0]):+.3f} "
                    f"vy={float(env.base_lin_vel[0, 1]):+.3f} "
                    f"z={float(env.base_pos[0, 2]):.3f} "
                    f"act_FR={float(action_abs[0:3].mean()):.3f} "
                    f"act_FL={float(action_abs[3:6].mean()):.3f} "
                    f"act_RR={float(action_abs[6:9].mean()):.3f} "
                    f"act_RL={float(action_abs[9:12].mean()):.3f}"
                )

            iteration += 1

            if dones.any():
                iteration = 0


if __name__ == "__main__":
    main()