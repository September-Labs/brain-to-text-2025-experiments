import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from b2txt.data import BrainToTextDataset, list_sessions, train_test_split_indices
from b2txt.metrics import normalize_sentence


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

    module = importlib.import_module(model_cfg["module"])
    model_cls = getattr(module, model_cfg["class"])

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


def load_predictions(pred_path: Path) -> List[Dict[str, object]]:
    entries = []
    with pred_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))
    return entries


def format_submission_rows(entries: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for entry in entries:
        if "pred_sentence" not in entry:
            raise ValueError("Missing pred_sentence in predictions. Enable decoding in the task.")
        rows.append(
            {
                "session": entry["session"],
                "block_num": int(entry["block_num"]),
                "trial_num": int(entry["trial_num"]),
                "text": normalize_sentence(entry["pred_sentence"]),
            }
        )
    rows.sort(key=lambda x: (x["session"], x["block_num"], x["trial_num"]))
    return rows


def write_submission(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for idx, row in enumerate(rows):
            writer.writerow([idx, row["text"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission CSV.")
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split to decode.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cpu, cuda:0).")
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Optional predictions JSONL from meta/evaluate.py.",
    )
    args = parser.parse_args()

    if args.predictions:
        entries = load_predictions(Path(args.predictions))
        rows = format_submission_rows(entries)
        write_submission(rows, Path(args.output))
        print(f"Wrote submission to {args.output} ({len(rows)} rows).")
        return

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    data_cfg = config["data"]
    training_cfg = dict(config["training"])
    if args.device:
        training_cfg["device"] = args.device
        if args.device.startswith("cpu"):
            training_cfg["require_cuda"] = False

    if not data_cfg.get("sessions"):
        data_cfg["sessions"] = list_sessions(data_cfg["dataset_dir"])

    device = select_device(training_cfg)
    model = build_model(config).to(device)
    task = build_task(config, device)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = Path(config["experiment"]["output_dir"]) / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    data_files = [
        str(Path(data_cfg["dataset_dir"]) / session / f"data_{args.split}.hdf5")
        for session in data_cfg["sessions"]
    ]
    _, split_trials = train_test_split_indices(
        data_files,
        test_percentage=1,
        seed=data_cfg.get("seed", -1),
        bad_trials_dict=data_cfg.get("bad_trials_dict"),
    )
    dataset = BrainToTextDataset(
        trial_indices=split_trials,
        n_batches=1,
        split="test",
        batch_size=data_cfg.get("val_batch_size", data_cfg["batch_size"]),
        days_per_batch=1,
        random_seed=data_cfg.get("seed", -1),
        feature_subset=data_cfg.get("feature_subset"),
        allow_unlabeled=True,
    )
    num_workers = data_cfg.get("num_workers", 0)
    prefetch_factor = data_cfg.get("prefetch_factor", 2)
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=bool(num_workers),
        prefetch_factor=prefetch_factor if num_workers else None,
    )

    model.eval()
    entries = []
    with torch.no_grad():
        for batch in loader:
            preds = task.predict_step(model, batch)
            for entry in task.format_predictions(batch, preds):
                entries.append(entry)

    rows = format_submission_rows(entries)
    write_submission(rows, Path(args.output))
    print(f"Wrote submission to {args.output} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
