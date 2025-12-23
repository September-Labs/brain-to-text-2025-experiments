import argparse
import importlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from b2txt.data import BrainToTextDataset, train_test_split_indices, list_sessions


def resolve_config_path(config_arg: str) -> Path:
    config_path = Path(config_arg)
    if config_path.exists():
        return config_path
    candidate = Path(__file__).parent / "configs" / config_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("b2txt.train")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s: %(message)s")

    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def select_device(training_cfg: Dict[str, object]) -> torch.device:
    device_str = training_cfg.get("device")
    if not device_str:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    require_cuda = bool(training_cfg.get("require_cuda", False))
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        if require_cuda:
            raise RuntimeError("CUDA requested but not available.")
        device_str = "cpu"
    return torch.device(device_str)


def build_model(config: Dict[str, object]) -> torch.nn.Module:
    model_cfg = config["model"]
    data_cfg = config["data"]

    module_path = model_cfg["module"]
    class_name = model_cfg["class"]

    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)

    params = dict(model_cfg.get("params", {}))
    params.setdefault("neural_dim", data_cfg.get("n_input_features", data_cfg.get("neural_dim")))
    params.setdefault("n_days", len(data_cfg["sessions"]))
    params.setdefault("n_classes", data_cfg["n_classes"])

    return model_cls(**params)


def build_task(config: Dict[str, object], device: torch.device):
    task_cfg = config["task"]
    module = importlib.import_module(task_cfg["module"])
    task_cls = getattr(module, task_cfg["class"])
    return task_cls(config, device)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: Dict[str, object],
) -> LambdaLR | None:
    scheduler_cfg = training_cfg.get("lr_scheduler")
    if not scheduler_cfg:
        return None

    if scheduler_cfg.get("type", "cosine") != "cosine":
        return None

    base_lr = training_cfg["lr"]
    min_lr = scheduler_cfg.get("min_lr", 0.0)
    warmup_steps = scheduler_cfg.get("warmup_steps", 0)
    decay_steps = scheduler_cfg.get("decay_steps", training_cfg["num_training_batches"])

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(max(1, decay_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        if base_lr == 0:
            return 1.0
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    tag: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: Dict[str, float],
) -> None:
    checkpoint = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / f"{tag}.pt")


def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    task,
) -> Dict[str, float]:
    model.eval()
    state = task.init_validation_state()
    with torch.no_grad():
        for batch in val_loader:
            stats = task.validation_step(model, batch)
            state = task.accumulate_validation(state, stats)
    return task.finalize_validation(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train brain-to-text models.")
    parser.add_argument("config", type=str, help="Path to YAML config.")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    training_cfg = config["training"]

    if not data_cfg.get("sessions"):
        data_cfg["sessions"] = list_sessions(data_cfg["dataset_dir"])

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Loaded config: %s", config_path)

    with (output_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)

    seed = config.get("experiment", {}).get("seed", -1)
    if seed != -1:
        torch.manual_seed(seed)

    device = select_device(training_cfg)
    logger.info("Using device: %s", device)

    model = build_model(config)
    model.to(device)
    task = build_task(config, device)

    if config.get("training", {}).get("compile", False) and hasattr(torch, "compile"):
        logger.info("Using torch.compile")
        model = torch.compile(model)

    train_files = [
        str(Path(data_cfg["dataset_dir"]) / session / "data_train.hdf5")
        for session in data_cfg["sessions"]
    ]
    val_files = [
        str(Path(data_cfg["dataset_dir"]) / session / "data_val.hdf5")
        for session in data_cfg["sessions"]
    ]

    train_trials, _ = train_test_split_indices(
        train_files,
        test_percentage=0,
        seed=data_cfg.get("seed", -1),
        bad_trials_dict=data_cfg.get("bad_trials_dict"),
    )
    _, val_trials = train_test_split_indices(
        val_files,
        test_percentage=1,
        seed=data_cfg.get("seed", -1),
        bad_trials_dict=data_cfg.get("bad_trials_dict"),
    )

    train_dataset = BrainToTextDataset(
        trial_indices=train_trials,
        n_batches=training_cfg["num_training_batches"],
        split="train",
        batch_size=data_cfg["batch_size"],
        days_per_batch=data_cfg.get("days_per_batch", 1),
        random_seed=data_cfg.get("seed", -1),
        must_include_days=data_cfg.get("must_include_days"),
        feature_subset=data_cfg.get("feature_subset"),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=data_cfg.get("loader_shuffle", False),
        num_workers=data_cfg.get("num_workers", 0),
    )

    val_dataset = BrainToTextDataset(
        trial_indices=val_trials,
        n_batches=1,
        split="test",
        batch_size=data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        days_per_batch=1,
        random_seed=data_cfg.get("seed", -1),
        feature_subset=data_cfg.get("feature_subset"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
        eps=training_cfg.get("epsilon", 1e-8),
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )
    scheduler = make_scheduler(optimizer, training_cfg)

    use_amp = training_cfg.get("use_amp", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step = 0
    resume_ckpt = training_cfg.get("resume_checkpoint")
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if training_cfg.get("resume_optimizer", False):
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = ckpt.get("step", 0)
        logger.info("Resumed from %s", resume_ckpt)

    metrics_path = output_dir / "metrics.jsonl"
    best_val_loss = float("inf")

    log_every = training_cfg.get("log_every_batches", 200)
    val_every = training_cfg.get("val_every_batches", 2000)
    save_every = training_cfg.get("save_every_batches", 0)

    model.train()

    for step, batch in enumerate(train_loader, start=1):
        global_step = start_step + step
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            step_output = task.training_step(model, batch)
            loss = step_output["loss"]

        scaler.scale(loss).backward()

        grad_clip = training_cfg.get("grad_clip_norm", 0.0)
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        if global_step % log_every == 0:
            logger.info("Step %d | train_loss=%.4f", global_step, loss.item())
            with metrics_path.open("a") as f:
                f.write(json.dumps({"step": global_step, "train_loss": loss.item()}) + "\n")

        if val_every and global_step % val_every == 0:
            metrics = run_validation(model, val_loader, task)
            logger.info("Step %d | val_metrics=%s", global_step, metrics)
            with metrics_path.open("a") as f:
                f.write(json.dumps({"step": global_step, **metrics}) + "\n")
            model.train()

            val_loss = metrics.get("val_loss")
            if (
                val_loss is not None
                and training_cfg.get("save_best", True)
                and val_loss < best_val_loss
            ):
                best_val_loss = val_loss
                save_checkpoint(
                    output_dir / "checkpoints",
                    "best",
                    model,
                    optimizer,
                    global_step,
                    metrics,
                )

        if save_every and global_step % save_every == 0:
            save_checkpoint(output_dir / "checkpoints", f"step_{global_step}", model, optimizer, global_step, {"train_loss": loss.item()})

    if training_cfg.get("save_last", True):
        save_checkpoint(output_dir / "checkpoints", "last", model, optimizer, start_step + len(train_loader), {"train_loss": loss.item()})

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
