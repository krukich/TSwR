import argparse
import os
import random

import torch
import torch.nn as nn


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


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = torch.load(path, map_location="cpu")

    required_keys = [
        "base_obs",
        "executed_actions",
        "actions",
        "privileged_encoded",
        "privileged_raw",
        "episode_ranges",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(f"Dataset is missing required key: {key}")

    return data


def validate_dataset_layout(data, action_key):
    metadata = data.get("metadata", {})

    base_num_obs = int(metadata.get("base_num_obs", data["base_obs"].shape[-1]))
    num_actions = int(metadata.get("num_actions", data[action_key].shape[-1]))
    privileged_encoder_dim = int(
        metadata.get("privileged_encoder_dim", data["privileged_encoded"].shape[-1])
    )
    privileged_raw_dim = int(
        metadata.get("privileged_raw_dim", data["privileged_raw"].shape[-1])
    )

    if base_num_obs != data["base_obs"].shape[-1]:
        raise RuntimeError(
            f"base_num_obs mismatch: metadata={base_num_obs}, "
            f"tensor={data['base_obs'].shape[-1]}"
        )

    if num_actions != data[action_key].shape[-1]:
        raise RuntimeError(
            f"num_actions mismatch: metadata={num_actions}, "
            f"tensor={data[action_key].shape[-1]}"
        )

    if privileged_encoder_dim != data["privileged_encoded"].shape[-1]:
        raise RuntimeError(
            f"privileged_encoder_dim mismatch: metadata={privileged_encoder_dim}, "
            f"tensor={data['privileged_encoded'].shape[-1]}"
        )

    if privileged_raw_dim != data["privileged_raw"].shape[-1]:
        raise RuntimeError(
            f"privileged_raw_dim mismatch: metadata={privileged_raw_dim}, "
            f"tensor={data['privileged_raw'].shape[-1]}"
        )

    if privileged_encoder_dim != 3 or privileged_raw_dim != 2:
        raise RuntimeError(
            "This trainer is for the fixed payload-position layout only: "
            f"expected privileged_encoder_dim=3 and privileged_raw_dim=2, "
            f"got encoded={privileged_encoder_dim}, raw={privileged_raw_dim}."
        )

    return {
        "metadata": metadata,
        "base_num_obs": base_num_obs,
        "num_actions": num_actions,
        "feature_dim": base_num_obs + num_actions,
        "privileged_encoder_dim": privileged_encoder_dim,
        "privileged_raw_dim": privileged_raw_dim,
    }


def filter_valid_episode_ranges(episode_ranges, window_len):
    lengths = episode_ranges[:, 1] - episode_ranges[:, 0]
    valid_mask = lengths >= window_len
    valid_ids = valid_mask.nonzero(as_tuple=False).flatten()

    if len(valid_ids) < 2:
        raise RuntimeError(
            f"Need at least 2 valid episodes with length >= window_len={window_len}. "
            f"Valid episodes found: {len(valid_ids)}."
        )

    return valid_ids


def split_episode_ids(valid_episode_ids, val_ratio):
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("--val_ratio must be in (0, 1)")

    num_episodes = len(valid_episode_ids)
    perm = valid_episode_ids[torch.randperm(num_episodes)]

    num_val = max(1, int(num_episodes * val_ratio))
    num_val = min(num_val, num_episodes - 1)

    val_episode_ids = perm[:num_val]
    train_episode_ids = perm[num_val:]

    return train_episode_ids, val_episode_ids


def build_window_indices(episode_ranges, window_len):
    all_indices = []

    for start, end in episode_ranges.tolist():
        length = end - start

        if length < window_len:
            continue

        indices = torch.arange(
            start + window_len - 1,
            end,
            dtype=torch.long,
        )

        all_indices.append(indices)

    if not all_indices:
        raise RuntimeError("No valid windows found. Try smaller --window_len.")

    return torch.cat(all_indices, dim=0)


def make_window_batch(features, targets, batch_end_indices, window_offsets):
    window_indices = batch_end_indices.unsqueeze(1) + window_offsets.unsqueeze(0)

    x = features[window_indices]
    x = x.reshape(batch_end_indices.shape[0], -1).float()

    y = targets[batch_end_indices].float()

    return x, y


def train_one_epoch_gpu(
    model,
    optimizer,
    loss_fn,
    features,
    targets,
    target_indices,
    window_offsets,
    batch_size,
    use_amp,
    scaler,
):
    model.train()

    perm = target_indices[
        torch.randperm(len(target_indices), device=target_indices.device)
    ]

    train_loss_sum = 0.0
    train_count = 0

    for start in range(0, len(perm), batch_size):
        end = min(start + batch_size, len(perm))
        batch_end_indices = perm[start:end]

        x, y = make_window_batch(
            features=features,
            targets=targets,
            batch_end_indices=batch_end_indices,
            window_offsets=window_offsets,
        )

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size_actual = batch_end_indices.shape[0]
        train_loss_sum += loss.item() * batch_size_actual
        train_count += batch_size_actual

    return train_loss_sum / max(train_count, 1)


def evaluate_gpu(
    model,
    features,
    targets,
    target_indices,
    window_offsets,
    batch_size,
    use_amp,
):
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    count = 0

    per_dim_abs_sum = None
    per_dim_sq_sum = None

    with torch.no_grad():
        for start in range(0, len(target_indices), batch_size):
            end = min(start + batch_size, len(target_indices))
            batch_end_indices = target_indices[start:end]

            x, y = make_window_batch(
                features=features,
                targets=targets,
                batch_end_indices=batch_end_indices,
                window_offsets=window_offsets,
            )

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(x)
            else:
                pred = model(x)

            diff = pred.float() - y.float()

            mse_sum += torch.sum(diff ** 2).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            count += y.numel()

            per_dim_abs = torch.sum(torch.abs(diff), dim=0).detach().cpu()
            per_dim_sq = torch.sum(diff ** 2, dim=0).detach().cpu()

            if per_dim_abs_sum is None:
                per_dim_abs_sum = per_dim_abs
                per_dim_sq_sum = per_dim_sq
            else:
                per_dim_abs_sum += per_dim_abs
                per_dim_sq_sum += per_dim_sq

    mse = mse_sum / max(count, 1)
    mae = mae_sum / max(count, 1)

    num_samples = len(target_indices)
    per_dim_mae = per_dim_abs_sum / max(num_samples, 1)
    per_dim_mse = per_dim_sq_sum / max(num_samples, 1)

    return mse, mae, per_dim_mae, per_dim_mse


def print_target_stats(data):
    y = data["privileged_encoded"].float()
    raw = data["privileged_raw"].float()

    print("============================================================")
    print("Target statistics")
    print("privileged_encoded target layout:")
    print("  dim 0: friction_linear")
    print("  dim 1: friction_log")
    print("  dim 2: payload_mass_encoded")
    print()
    print(f"encoded mean: {torch.mean(y, dim=0).numpy()}")
    print(f"encoded std:  {torch.std(y, dim=0).numpy()}")
    print(f"encoded min:  {torch.min(y, dim=0).values.numpy()}")
    print(f"encoded max:  {torch.max(y, dim=0).values.numpy()}")
    print()
    print("privileged_raw layout:")
    print("  dim 0: ground_friction")
    print("  dim 1: payload_mass")
    print()
    print(f"raw mean: {torch.mean(raw, dim=0).numpy()}")
    print(f"raw std:  {torch.std(raw, dim=0).numpy()}")
    print(f"raw min:  {torch.min(raw, dim=0).values.numpy()}")
    print(f"raw max:  {torch.max(raw, dim=0).values.numpy()}")
    print("============================================================")


def load_initial_weights(
    model,
    init_from,
    expected_input_dim,
    expected_output_dim,
    expected_window_len,
    expected_action_key,
    device,
):
    if init_from is None:
        return

    if not os.path.exists(init_from):
        raise FileNotFoundError(f"Initial adapter checkpoint not found: {init_from}")

    checkpoint = torch.load(init_from, map_location=device)

    ckpt_input_dim = int(checkpoint["input_dim"])
    ckpt_output_dim = int(checkpoint["output_dim"])
    ckpt_window_len = int(checkpoint.get("window_len", -1))
    ckpt_action_key = checkpoint.get("action_key", "unknown")

    if ckpt_input_dim != expected_input_dim:
        raise RuntimeError(
            f"init_from input_dim mismatch: checkpoint={ckpt_input_dim}, "
            f"expected={expected_input_dim}"
        )

    if ckpt_output_dim != expected_output_dim:
        raise RuntimeError(
            f"init_from output_dim mismatch: checkpoint={ckpt_output_dim}, "
            f"expected={expected_output_dim}"
        )

    if ckpt_window_len != expected_window_len:
        raise RuntimeError(
            f"init_from window_len mismatch: checkpoint={ckpt_window_len}, "
            f"expected={expected_window_len}"
        )

    if ckpt_action_key != expected_action_key:
        print(
            "[warning] init_from action_key differs from current action_key: "
            f"checkpoint={ckpt_action_key}, current={expected_action_key}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("============================================================")
    print(f"Initialized adapter from: {init_from}")
    print(f"checkpoint best_epoch: {checkpoint.get('best_epoch', 'unknown')}")
    print(f"checkpoint best_val_mse: {checkpoint.get('best_val_mse', 'unknown')}")
    print("============================================================")


def save_checkpoint(
    path,
    model,
    args,
    dims,
    metadata,
    target_names,
    best_val_mse,
    best_epoch,
    train_episode_ids,
    val_episode_ids,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "AdaptationMLP",
        "input_dim": args.window_len * dims["feature_dim"],
        "output_dim": dims["privileged_encoder_dim"],
        "window_len": args.window_len,
        "action_key": args.action_key,
        "base_num_obs": dims["base_num_obs"],
        "num_actions": dims["num_actions"],
        "feature_dim": dims["feature_dim"],
        "privileged_encoder_dim": dims["privileged_encoder_dim"],
        "privileged_raw_dim": dims["privileged_raw_dim"],
        "target_names": target_names,
        "best_val_mse": best_val_mse,
        "best_epoch": best_epoch,
        "dataset_path": args.data,
        "dataset_metadata": metadata,
        "train_episode_ids": train_episode_ids.cpu(),
        "val_episode_ids": val_episode_ids.cpu(),
        "fixed_payload_position_layout": True,
        "trainer": "gpu_window_indexing",
        "init_from": args.init_from,
    }

    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)

    parser.add_argument(
        "--out",
        type=str,
        default="logs/adaptation/adaptation_mlp_h30_fixed_payload.pt",
    )

    parser.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Optional adapter checkpoint to initialize weights from.",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--window_len", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument(
        "--action_key",
        type=str,
        default="executed_actions",
        choices=["actions", "executed_actions"],
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision training on CUDA.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    if args.amp and device.type != "cuda":
        raise RuntimeError("--amp can only be used with CUDA")

    eval_batch_size = args.eval_batch_size
    if eval_batch_size is None:
        eval_batch_size = args.batch_size

    print("============================================================")
    print("Training adaptation module")
    print(f"Dataset: {args.data}")
    print(f"Output: {args.out}")
    print(f"Init from: {args.init_from}")
    print(f"Device: {device}")
    print(f"Window length: {args.window_len}")
    print(f"Action key: {args.action_key}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"AMP: {args.amp}")
    print(f"Seed: {args.seed}")
    print("============================================================")

    if args.action_key == "actions":
        print(
            "[warning] action_key='actions' means current selected actor actions. "
            "For causal RMA-style adaptation, 'executed_actions' is usually better."
        )

    data = load_dataset(args.data)
    dims = validate_dataset_layout(data, args.action_key)

    metadata = dims["metadata"]

    input_dim = args.window_len * dims["feature_dim"]
    output_dim = dims["privileged_encoder_dim"]

    print(f"base_num_obs: {dims['base_num_obs']}")
    print(f"num_actions: {dims['num_actions']}")
    print(f"feature_dim: {dims['feature_dim']}")
    print(f"input_dim: {input_dim}")
    print(f"output_dim: {output_dim}")
    print(f"privileged_raw_dim: {dims['privileged_raw_dim']}")
    print(f"episodes total: {len(data['episode_ranges'])}")
    print(f"samples total: {len(data['base_obs'])}")

    print_target_stats(data)

    episode_ranges = data["episode_ranges"]

    valid_episode_ids = filter_valid_episode_ranges(
        episode_ranges=episode_ranges,
        window_len=args.window_len,
    )

    train_episode_ids, val_episode_ids = split_episode_ids(
        valid_episode_ids=valid_episode_ids,
        val_ratio=args.val_ratio,
    )

    train_ranges = episode_ranges[train_episode_ids]
    val_ranges = episode_ranges[val_episode_ids]

    train_target_indices = build_window_indices(
        train_ranges,
        args.window_len,
    )

    val_target_indices = build_window_indices(
        val_ranges,
        args.window_len,
    )

    print("============================================================")
    print(f"valid episodes: {len(valid_episode_ids)}")
    print(f"train episodes: {len(train_ranges)}")
    print(f"val episodes: {len(val_ranges)}")
    print(f"train windows: {len(train_target_indices)}")
    print(f"val windows: {len(val_target_indices)}")
    print("============================================================")

    print("Moving tensors to GPU / device...")

    features = torch.cat(
        [
            data["base_obs"],
            data[args.action_key],
        ],
        dim=-1,
    ).to(device=device, non_blocking=True)

    targets = data["privileged_encoded"].to(device=device, non_blocking=True)

    train_target_indices = train_target_indices.to(device=device, non_blocking=True)
    val_target_indices = val_target_indices.to(device=device, non_blocking=True)

    window_offsets = torch.arange(
        -(args.window_len - 1),
        1,
        device=device,
        dtype=torch.long,
    )

    del data

    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = AdaptationMLP(
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)

    load_initial_weights(
        model=model,
        init_from=args.init_from,
        expected_input_dim=input_dim,
        expected_output_dim=output_dim,
        expected_window_len=args.window_len,
        expected_action_key=args.action_key,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_mse = float("inf")
    best_epoch = 0

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    target_names = [
        "friction_linear",
        "friction_log",
        "payload_mass_encoded",
    ]

    for epoch in range(1, args.epochs + 1):
        train_mse = train_one_epoch_gpu(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            features=features,
            targets=targets,
            target_indices=train_target_indices,
            window_offsets=window_offsets,
            batch_size=args.batch_size,
            use_amp=args.amp,
            scaler=scaler,
        )

        val_mse, val_mae, per_dim_mae, per_dim_mse = evaluate_gpu(
            model=model,
            features=features,
            targets=targets,
            target_indices=val_target_indices,
            window_offsets=window_offsets,
            batch_size=eval_batch_size,
            use_amp=args.amp,
        )

        per_dim_text = ", ".join(
            f"{name}={value:.6f}"
            for name, value in zip(target_names, per_dim_mae.tolist())
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_mse={train_mse:.6f} | "
            f"val_mse={val_mse:.6f} | "
            f"val_mae={val_mae:.6f} | "
            f"per_dim_mae=[{per_dim_text}]"
        )

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch

            save_checkpoint(
                path=args.out,
                model=model,
                args=args,
                dims=dims,
                metadata=metadata,
                target_names=target_names,
                best_val_mse=best_val_mse,
                best_epoch=best_epoch,
                train_episode_ids=train_episode_ids,
                val_episode_ids=val_episode_ids,
            )

            print(f"Saved best model to: {args.out}")

    print("============================================================")
    print(f"Finished. Best epoch: {best_epoch}")
    print(f"Best val_mse: {best_val_mse:.6f}")
    print(f"Best model: {args.out}")
    print("============================================================")


if __name__ == "__main__":
    main()