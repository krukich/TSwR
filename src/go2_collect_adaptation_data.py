import argparse
import os
import pickle

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


def gs_backend(device: str):
    if device == "cuda:0":
        return gs.constants.backend.gpu
    return gs.constants.backend.cpu


def load_cfgs(log_dir: str):
    cfg_path = os.path.join(log_dir, "cfgs.pkl")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "rb") as f:
        return pickle.load(f)


def validate_layout(obs_cfg, env_cfg, command_cfg):
    base_num_obs = int(obs_cfg["base_num_obs"])
    privileged_encoder_dim = int(obs_cfg["privileged_encoder_dim"])
    privileged_raw_dim = int(obs_cfg["privileged_raw_dim"])

    num_obs = int(obs_cfg["num_obs"])
    num_privileged_obs = int(obs_cfg["num_privileged_obs"])
    num_actions = int(env_cfg["num_actions"])
    num_commands = int(command_cfg["num_commands"])

    expected_num_obs = base_num_obs + privileged_encoder_dim
    expected_num_privileged_obs = expected_num_obs + privileged_raw_dim

    if num_obs != expected_num_obs:
        raise RuntimeError(
            f"Bad obs layout: num_obs={num_obs}, expected={expected_num_obs}. "
            "Probably cfgs.pkl was created with another observation layout."
        )

    if num_privileged_obs != expected_num_privileged_obs:
        raise RuntimeError(
            f"Bad privileged obs layout: num_privileged_obs={num_privileged_obs}, "
            f"expected={expected_num_privileged_obs}."
        )

    if privileged_raw_dim != 2 or privileged_encoder_dim != 3:
        raise RuntimeError(
            "This collector expects fixed payload-position layout: "
            "privileged_raw_dim=2 and privileged_encoder_dim=3. "
            f"Got raw={privileged_raw_dim}, encoded={privileged_encoder_dim}."
        )

    return {
        "base_num_obs": base_num_obs,
        "privileged_encoder_dim": privileged_encoder_dim,
        "privileged_raw_dim": privileged_raw_dim,
        "num_obs": num_obs,
        "num_privileged_obs": num_privileged_obs,
        "num_actions": num_actions,
        "num_commands": num_commands,
    }


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


def make_policy_obs(base_obs, z):
    return torch.cat([base_obs, z], dim=-1)


def sort_by_episode(data, sample_count, max_episode_length):
    episode_id = data["episode_id"][:sample_count]
    saved_step = data["saved_step_in_episode"][:sample_count]

    key = episode_id * (max_episode_length + 1) + saved_step
    perm = torch.argsort(key)

    sorted_data = {}

    for name, value in data.items():
        if name == "metadata":
            sorted_data[name] = value
        else:
            sorted_data[name] = value[:sample_count][perm]

    sorted_episode_id = sorted_data["episode_id"]

    unique_episode_ids, counts = torch.unique_consecutive(
        sorted_episode_id,
        return_counts=True,
    )

    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts

    sorted_data["episode_ids"] = unique_episode_ids
    sorted_data["episode_ranges"] = torch.stack([starts, ends], dim=1)

    return sorted_data


def allocate_dataset(max_samples, dims):
    def empty(shape, dtype=torch.float16):
        return torch.empty(shape, dtype=dtype, device="cpu")

    return {
        "obs": empty((max_samples, dims["num_obs"])),

        "teacher_obs": empty((max_samples, dims["num_obs"])),

        "base_obs": empty((max_samples, dims["base_num_obs"])),

        "actions": empty((max_samples, dims["num_actions"])),

        "executed_actions": empty((max_samples, dims["num_actions"])),

        "privileged_obs": empty((max_samples, dims["num_privileged_obs"])),
        "privileged_encoded": empty((max_samples, dims["privileged_encoder_dim"])),
        "privileged_raw": empty((max_samples, dims["privileged_raw_dim"])),

        "predicted_privileged_encoded": empty(
            (max_samples, dims["privileged_encoder_dim"])
        ),

        "policy_privileged_encoded": empty(
            (max_samples, dims["privileged_encoder_dim"])
        ),

        "commands": empty((max_samples, dims["num_commands"])),

        "used_true_privileged": empty((max_samples,), dtype=torch.bool),
        "episode_id": empty((max_samples,), dtype=torch.long),
        "saved_step_in_episode": empty((max_samples,), dtype=torch.long),
        "sim_step_in_episode": empty((max_samples,), dtype=torch.long),
    }


def copy_rows(dst, src, ids, dst_slice, dtype=torch.float16):
    dst[dst_slice].copy_(
        src[ids].detach().to("cpu", dtype=dtype)
    )


def print_header(args, dims, ckpt_path, adaptation_model, window_len, action_key):
    print("============================================================")
    print("Collecting adaptation data")
    print(f"Experiment: {args.exp_name}")
    print(f"Teacher checkpoint: {ckpt_path}")
    print(f"rollout_mode: {args.rollout_mode}")
    print(f"adaptation_path: {args.adaptation_path}")
    print(f"teacher_prob: {args.teacher_prob}")
    print(f"bootstrap_true_steps: {args.bootstrap_true_steps}")
    print(f"num_envs: {args.num_envs}")
    print(f"max_samples: {args.max_samples}")
    print(f"skip_initial_steps: {args.skip_initial_steps}")
    print(f"base_num_obs: {dims['base_num_obs']}")
    print(f"num_obs: {dims['num_obs']}")
    print(f"num_privileged_obs: {dims['num_privileged_obs']}")
    print(f"privileged_encoder_dim: {dims['privileged_encoder_dim']}")
    print(f"privileged_raw_dim: {dims['privileged_raw_dim']}")

    if adaptation_model is not None:
        print(f"adapter window_len: {window_len}")
        print(f"adapter action_key: {action_key}")

    print("============================================================")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("-B", "--num_envs", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"])

    parser.add_argument("--max_samples", type=int, default=300_000)
    parser.add_argument("--skip_initial_steps", type=int, default=150)

    parser.add_argument(
        "--rollout_mode",
        type=str,
        default="teacher",
        choices=["teacher", "adapted", "mixed"],
    )

    parser.add_argument("--adaptation_path", type=str, default=None)

    parser.add_argument(
        "--teacher_prob",
        type=float,
        default=0.5,
        help="Only for mixed rollout. Probability of using true privileged z.",
    )

    parser.add_argument(
        "--bootstrap_true_steps",
        type=int,
        default=150,
        help="First N steps of each episode use true privileged z in adapted/mixed mode.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="data/adaptation/go2_teacher_rollout_300k_skip150.pt",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rollout_mode in ["adapted", "mixed"] and args.adaptation_path is None:
        raise ValueError("--adaptation_path is required for adapted/mixed rollout")

    if not (0.0 <= args.teacher_prob <= 1.0):
        raise ValueError("--teacher_prob must be in [0, 1]")

    gs.init(
        logging_level="warning",
        backend=gs_backend(args.device),
    )

    device = torch.device(args.device)

    log_dir = os.path.join("logs", args.exp_name)
    ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_cfgs(log_dir)
    dims = validate_layout(obs_cfg, env_cfg, command_cfg)

    base_num_obs = dims["base_num_obs"]
    privileged_encoder_dim = dims["privileged_encoder_dim"]
    privileged_raw_dim = dims["privileged_raw_dim"]
    num_obs = dims["num_obs"]
    num_actions = dims["num_actions"]

    local_reward_cfg = dict(reward_cfg)
    local_reward_cfg["reward_scales"] = {}

    adaptation_model = None
    adaptation_ckpt = None
    window_len = 1
    action_key = "executed_actions"

    if args.rollout_mode in ["adapted", "mixed"]:
        adaptation_model, adaptation_ckpt, window_len, action_key = load_adaptation_model(
            args.adaptation_path,
            device,
        )

        expected_input_dim = window_len * (base_num_obs + num_actions)

        if int(adaptation_ckpt["input_dim"]) != expected_input_dim:
            raise RuntimeError(
                f"Bad adaptation input_dim: "
                f"checkpoint={adaptation_ckpt['input_dim']}, expected={expected_input_dim}"
            )

        if int(adaptation_ckpt["output_dim"]) != privileged_encoder_dim:
            raise RuntimeError(
                f"Bad adaptation output_dim: "
                f"checkpoint={adaptation_ckpt['output_dim']}, "
                f"expected={privileged_encoder_dim}"
            )

        if action_key not in ["executed_actions", "actions"]:
            raise RuntimeError(
                f"Unsupported adapter action_key='{action_key}'. "
                "Expected 'executed_actions' or 'actions'."
            )

    print_header(
        args=args,
        dims=dims,
        ckpt_path=ckpt_path,
        adaptation_model=adaptation_model,
        window_len=window_len,
        action_key=action_key,
    )

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=local_reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
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
        (args.num_envs, window_len, base_num_obs),
        device=device,
        dtype=torch.float32,
    )

    history_actions = torch.zeros(
        (args.num_envs, window_len, num_actions),
        device=device,
        dtype=torch.float32,
    )

    data = allocate_dataset(args.max_samples, dims)

    obs, privileged_obs = env.reset()

    episode_step = torch.zeros(
        (args.num_envs,),
        dtype=torch.long,
        device=device,
    )

    episode_id = torch.arange(
        args.num_envs,
        dtype=torch.long,
        device=device,
    )

    next_episode_id = args.num_envs
    sample_count = 0
    total_sim_steps = 0

    z_mse_sum = 0.0
    z_mae_sum = 0.0
    z_count = 0

    with torch.no_grad():
        while sample_count < args.max_samples:
            base_obs = obs[:, :base_num_obs]
            z_true = obs[:, base_num_obs:base_num_obs + privileged_encoder_dim]
            privileged_raw = privileged_obs[:, num_obs:num_obs + privileged_raw_dim]

            previous_action = env.last_actions.clone()

            history_base_obs = torch.roll(history_base_obs, shifts=-1, dims=1)
            history_actions = torch.roll(history_actions, shifts=-1, dims=1)

            history_base_obs[:, -1, :] = base_obs
            history_actions[:, -1, :] = previous_action

            if adaptation_model is not None:
                adaptation_input = torch.cat(
                    [history_base_obs, history_actions],
                    dim=-1,
                ).reshape(args.num_envs, -1)

                z_pred = adaptation_model(adaptation_input)
                z_pred = torch.clamp(z_pred, -1.0, 1.0)
            else:
                z_pred = z_true.clone()

            if args.rollout_mode == "teacher":
                use_true_mask = torch.ones(
                    (args.num_envs,),
                    dtype=torch.bool,
                    device=device,
                )
            elif args.rollout_mode == "adapted":
                use_true_mask = episode_step < args.bootstrap_true_steps
            else:
                bootstrap_mask = episode_step < args.bootstrap_true_steps
                random_true_mask = (
                    torch.rand((args.num_envs,), device=device) < args.teacher_prob
                )
                use_true_mask = bootstrap_mask | random_true_mask

            z_policy = torch.where(
                use_true_mask.unsqueeze(-1),
                z_true,
                z_pred,
            )

            obs_for_policy = make_policy_obs(base_obs, z_policy)
            teacher_obs = make_policy_obs(base_obs, z_true)

            actions = policy(obs_for_policy)

            if env.simulate_action_latency:
                executed_actions = env.last_actions.clone()
            else:
                executed_actions = actions.clone()

            eligible_mask = episode_step >= args.skip_initial_steps
            eligible_ids = eligible_mask.nonzero(as_tuple=False).flatten()

            if len(eligible_ids) > 0:
                remaining = args.max_samples - sample_count
                take = min(len(eligible_ids), remaining)

                ids = eligible_ids[:take]
                dst = slice(sample_count, sample_count + take)

                copy_rows(data["obs"], obs_for_policy, ids, dst)
                copy_rows(data["teacher_obs"], teacher_obs, ids, dst)
                copy_rows(data["base_obs"], base_obs, ids, dst)
                copy_rows(data["actions"], actions, ids, dst)
                copy_rows(data["executed_actions"], executed_actions, ids, dst)
                copy_rows(data["privileged_obs"], privileged_obs, ids, dst)
                copy_rows(data["privileged_encoded"], z_true, ids, dst)
                copy_rows(data["privileged_raw"], privileged_raw, ids, dst)
                copy_rows(data["predicted_privileged_encoded"], z_pred, ids, dst)
                copy_rows(data["policy_privileged_encoded"], z_policy, ids, dst)
                copy_rows(data["commands"], env.commands, ids, dst)

                data["used_true_privileged"][dst].copy_(
                    use_true_mask[ids].detach().to("cpu")
                )
                data["episode_id"][dst].copy_(
                    episode_id[ids].detach().to("cpu")
                )
                data["sim_step_in_episode"][dst].copy_(
                    episode_step[ids].detach().to("cpu")
                )
                data["saved_step_in_episode"][dst].copy_(
                    (episode_step[ids] - args.skip_initial_steps).detach().to("cpu")
                )

                if adaptation_model is not None:
                    diff = z_pred[ids] - z_true[ids]
                    z_mse_sum += torch.sum(diff ** 2).item()
                    z_mae_sum += torch.sum(torch.abs(diff)).item()
                    z_count += diff.numel()

                sample_count += take

                if sample_count % 10_000 < take:
                    if z_count > 0:
                        z_mse = z_mse_sum / z_count
                        z_mae = z_mae_sum / z_count
                    else:
                        z_mse = 0.0
                        z_mae = 0.0

                    true_ratio = float(torch.mean(use_true_mask[ids].float()).item())

                    print(
                        f"Collected {sample_count}/{args.max_samples} samples | "
                        f"sim_steps={total_sim_steps} | "
                        f"true_ratio_saved={true_ratio:.3f} | "
                        f"adapter_z_mse={z_mse:.6f} | "
                        f"adapter_z_mae={z_mae:.6f}"
                    )

            obs, privileged_obs, rewards, dones, infos = env.step(
                actions,
                is_train=True,
            )

            done_ids = dones.nonzero(as_tuple=False).flatten()
            alive_ids = (~dones.bool()).nonzero(as_tuple=False).flatten()

            episode_step[alive_ids] += 1

            if len(done_ids) > 0:
                num_done = len(done_ids)

                episode_step[done_ids] = 0
                episode_id[done_ids] = torch.arange(
                    next_episode_id,
                    next_episode_id + num_done,
                    dtype=torch.long,
                    device=device,
                )

                history_base_obs[done_ids] = 0.0
                history_actions[done_ids] = 0.0

                next_episode_id += num_done

            total_sim_steps += 1

    data["metadata"] = {
        "exp_name": args.exp_name,
        "teacher_ckpt": args.ckpt,
        "rollout_mode": args.rollout_mode,
        "adaptation_path": args.adaptation_path,
        "teacher_prob": args.teacher_prob,
        "bootstrap_true_steps": args.bootstrap_true_steps,
        "num_envs": args.num_envs,
        "max_samples": args.max_samples,
        "skip_initial_steps": args.skip_initial_steps,
        "total_sim_steps": total_sim_steps,
        "base_num_obs": dims["base_num_obs"],
        "num_obs": dims["num_obs"],
        "num_privileged_obs": dims["num_privileged_obs"],
        "num_actions": dims["num_actions"],
        "num_commands": dims["num_commands"],
        "privileged_encoder_dim": dims["privileged_encoder_dim"],
        "privileged_raw_dim": dims["privileged_raw_dim"],
        "adapter_window_len_used_for_rollout": window_len,
        "action_key": action_key,
        "target": "privileged_encoded",
        "adapter_input": "history of base_obs and previous actions",
        "format": (
            "Flat tensors sorted by episode_id and saved_step_in_episode. "
            "episode_ranges contains [start, end) indices for every saved episode segment. "
            "First skip_initial_steps simulator steps of every episode are not saved. "
            "obs is the actual actor input during rollout: base_obs + policy_privileged_encoded. "
            "teacher_obs is base_obs + true privileged_encoded. "
            "base_obs contains no privileged information. "
            "privileged_encoded is the true adapter target. "
            "privileged_raw contains [ground_friction, payload_mass]. "
            "predicted_privileged_encoded is the adapter prediction during rollout. "
            "policy_privileged_encoded is the privileged vector actually passed to the actor."
        ),
    }

    sorted_data = sort_by_episode(
        data=data,
        sample_count=sample_count,
        max_episode_length=int(env.max_episode_length),
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(sorted_data, args.out)

    print("============================================================")
    print(f"Saved dataset to: {args.out}")
    print(f"Samples: {sample_count}")
    print(f"Episodes with saved data: {len(sorted_data['episode_ranges'])}")
    print(f"obs: {tuple(sorted_data['obs'].shape)}")
    print(f"teacher_obs: {tuple(sorted_data['teacher_obs'].shape)}")
    print(f"base_obs: {tuple(sorted_data['base_obs'].shape)}")
    print(f"actions: {tuple(sorted_data['actions'].shape)}")
    print(f"executed_actions: {tuple(sorted_data['executed_actions'].shape)}")
    print(f"privileged_encoded: {tuple(sorted_data['privileged_encoded'].shape)}")
    print(f"privileged_raw: {tuple(sorted_data['privileged_raw'].shape)}")
    print(f"predicted_privileged_encoded: {tuple(sorted_data['predicted_privileged_encoded'].shape)}")
    print(f"policy_privileged_encoded: {tuple(sorted_data['policy_privileged_encoded'].shape)}")
    print(f"episode_ranges: {tuple(sorted_data['episode_ranges'].shape)}")
    print("============================================================")


if __name__ == "__main__":
    main()