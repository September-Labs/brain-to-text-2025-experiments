import os
import math
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class BrainToTextDataset(Dataset):
    """Dataset for brain-to-text HDF5 data.

    Returns full batches rather than single examples.
    """

    def __init__(
        self,
        trial_indices: Dict[int, Dict[str, object]],
        n_batches: int,
        split: str = "train",
        batch_size: int = 64,
        days_per_batch: int = 1,
        random_seed: int = -1,
        must_include_days: List[int] | None = None,
        feature_subset: List[int] | None = None,
        allow_unlabeled: bool = False,
    ) -> None:
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.split = split
        self.days_per_batch = days_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.trial_indices = trial_indices
        self.n_days = len(trial_indices.keys())
        self.feature_subset = feature_subset
        self.allow_unlabeled = allow_unlabeled

        self.n_trials = 0
        for day in trial_indices:
            self.n_trials += len(trial_indices[day]["trials"])

        if must_include_days is not None and len(must_include_days) > days_per_batch:
            raise ValueError(
                "must_include_days must be <= days_per_batch "
                f"(got {must_include_days}, days_per_batch={days_per_batch})"
            )

        if must_include_days is not None and len(must_include_days) > self.n_days and split != "train":
            raise ValueError(
                "must_include_days is only valid for training data. "
                f"Got {must_include_days} with {self.n_days} days"
            )

        if must_include_days is not None:
            for i, day in enumerate(must_include_days):
                if day < 0:
                    must_include_days[i] = self.n_days + day

        self.must_include_days = must_include_days

        if self.split == "train" and self.days_per_batch > self.n_days:
            raise ValueError(
                f"days_per_batch {days_per_batch} exceeds available days {self.n_days}"
            )

        if self.split == "train":
            self.batch_index = self._create_batch_index_train()
        else:
            self.batch_index = self._create_batch_index_test()
            self.n_batches = len(self.batch_index.keys())

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            "input_features": [],
            "seq_class_ids": [],
            "n_time_steps": [],
            "phone_seq_lens": [],
            "day_indicies": [],
            "transcriptions": [],
            "sentence_labels": [],
            "block_nums": [],
            "trial_nums": [],
        }

        index = self.batch_index[idx]

        for day in index.keys():
            session_path = self.trial_indices[day]["session_path"]
            if not os.path.exists(session_path):
                continue

            with h5py.File(session_path, "r") as f:
                for trial in index[day]:
                    try:
                        group = f[f"trial_{trial:04d}"]
                        input_features = torch.from_numpy(group["input_features"][:])
                        if self.feature_subset:
                            input_features = input_features[:, self.feature_subset]

                        batch["input_features"].append(input_features)

                        if "seq_class_ids" in group:
                            seq_class_ids = torch.from_numpy(group["seq_class_ids"][:])
                            seq_len = group.attrs.get("seq_len", 0)
                        elif self.allow_unlabeled:
                            seq_class_ids = torch.zeros(1, dtype=torch.int64)
                            seq_len = 0
                        else:
                            raise KeyError("seq_class_ids not found")

                        if "transcription" in group:
                            transcription = torch.from_numpy(group["transcription"][:])
                        elif self.allow_unlabeled:
                            transcription = torch.zeros(1, dtype=torch.int16)
                        else:
                            raise KeyError("transcription not found")

                        if "sentence_label" in group.attrs:
                            sentence_label = group.attrs["sentence_label"]
                        elif self.allow_unlabeled:
                            sentence_label = ""
                        else:
                            raise KeyError("sentence_label not found")

                        batch["seq_class_ids"].append(seq_class_ids)
                        batch["transcriptions"].append(transcription)
                        batch["n_time_steps"].append(group.attrs["n_time_steps"])
                        batch["phone_seq_lens"].append(seq_len)
                        batch["day_indicies"].append(int(day))
                        batch["sentence_labels"].append(sentence_label)
                        batch["block_nums"].append(group.attrs["block_num"])
                        batch["trial_nums"].append(group.attrs["trial_num"])
                    except Exception:
                        continue

        if not batch["input_features"]:
            raise RuntimeError(f"Empty batch at idx {idx} (check dataset paths).")

        batch["input_features"] = pad_sequence(
            batch["input_features"], batch_first=True, padding_value=0
        )
        batch["seq_class_ids"] = pad_sequence(
            batch["seq_class_ids"], batch_first=True, padding_value=0
        )

        batch["n_time_steps"] = torch.tensor(batch["n_time_steps"])
        batch["phone_seq_lens"] = torch.tensor(batch["phone_seq_lens"])
        batch["day_indicies"] = torch.tensor(batch["day_indicies"])
        batch["transcriptions"] = torch.stack(batch["transcriptions"])
        batch["block_nums"] = torch.tensor(batch["block_nums"])
        batch["trial_nums"] = torch.tensor(batch["trial_nums"])

        return batch

    def _create_batch_index_train(self) -> Dict[int, Dict[int, np.ndarray]]:
        batch_index = {}

        if self.must_include_days is not None:
            non_must = [d for d in self.trial_indices.keys() if d not in self.must_include_days]

        for batch_idx in range(self.n_batches):
            batch = {}

            if self.must_include_days is not None and len(self.must_include_days) > 0:
                days = np.concatenate(
                    (
                        self.must_include_days,
                        np.random.choice(
                            non_must,
                            size=self.days_per_batch - len(self.must_include_days),
                            replace=False,
                        ),
                    )
                )
            else:
                days = np.random.choice(
                    list(self.trial_indices.keys()),
                    size=self.days_per_batch,
                    replace=False,
                )

            num_trials = math.ceil(self.batch_size / self.days_per_batch)

            for day in days:
                trials = np.random.choice(
                    self.trial_indices[day]["trials"], size=num_trials, replace=True
                )
                batch[day] = trials

            extra_trials = (num_trials * len(days)) - self.batch_size
            while extra_trials > 0:
                day = np.random.choice(days)
                batch[day] = batch[day][:-1]
                extra_trials -= 1

            batch_index[batch_idx] = batch

        return batch_index

    def _create_batch_index_test(self) -> Dict[int, Dict[int, List[int]]]:
        batch_index = {}
        batch_idx = 0

        for day in self.trial_indices.keys():
            num_trials = len(self.trial_indices[day]["trials"])
            num_batches = (num_trials + self.batch_size - 1) // self.batch_size

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_trials)
                batch_trials = self.trial_indices[day]["trials"][start_idx:end_idx]
                batch_index[batch_idx] = {day: batch_trials}
                batch_idx += 1

        return batch_index


def train_test_split_indices(
    file_paths: List[str],
    test_percentage: float = 0.1,
    seed: int = -1,
    bad_trials_dict: Dict[str, Dict[str, List[int]]] | None = None,
) -> Tuple[Dict[int, Dict[str, object]], Dict[int, Dict[str, object]]]:
    """Split each session file into train/test indices."""
    if seed != -1:
        np.random.seed(seed)

    trials_per_day: Dict[int, Dict[str, object]] = {}
    for idx, path in enumerate(file_paths):
        session = [s for s in path.split("/") if s.startswith("t15.") or s.startswith("t12.")]
        session = session[0] if session else f"session_{idx}"

        good_trial_indices = []
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                num_trials = len(list(f.keys()))
                for trial in range(num_trials):
                    key = f"trial_{trial:04d}"
                    block_num = f[key].attrs["block_num"]
                    trial_num = f[key].attrs["trial_num"]

                    if (
                        bad_trials_dict
                        and session in bad_trials_dict
                        and str(block_num) in bad_trials_dict[session]
                        and trial_num in bad_trials_dict[session][str(block_num)]
                    ):
                        continue

                    good_trial_indices.append(trial)

        trials_per_day[idx] = {
            "num_trials": len(good_trial_indices),
            "trial_indices": good_trial_indices,
            "session_path": path,
        }

    train_trials: Dict[int, Dict[str, object]] = {}
    test_trials: Dict[int, Dict[str, object]] = {}

    for day in trials_per_day.keys():
        num_trials = trials_per_day[day]["num_trials"]
        all_indices = trials_per_day[day]["trial_indices"]

        if test_percentage == 0:
            train_trials[day] = {
                "trials": all_indices,
                "session_path": trials_per_day[day]["session_path"],
            }
            test_trials[day] = {"trials": [], "session_path": trials_per_day[day]["session_path"]}
            continue

        if test_percentage == 1:
            train_trials[day] = {"trials": [], "session_path": trials_per_day[day]["session_path"]}
            test_trials[day] = {
                "trials": all_indices,
                "session_path": trials_per_day[day]["session_path"],
            }
            continue

        if num_trials == 0:
            train_trials[day] = {"trials": [], "session_path": trials_per_day[day]["session_path"]}
            test_trials[day] = {"trials": [], "session_path": trials_per_day[day]["session_path"]}
            continue

        num_test = max(1, int(num_trials * test_percentage))
        test_indices = np.random.choice(all_indices, size=num_test, replace=False).tolist()
        train_indices = [idx for idx in all_indices if idx not in test_indices]

        train_trials[day] = {
            "trials": train_indices,
            "session_path": trials_per_day[day]["session_path"],
        }
        test_trials[day] = {
            "trials": test_indices,
            "session_path": trials_per_day[day]["session_path"],
        }

    return train_trials, test_trials


def list_sessions(dataset_dir: str) -> List[str]:
    """List session directories under the dataset directory."""
    if not os.path.isdir(dataset_dir):
        return []
    sessions = [name for name in os.listdir(dataset_dir) if name.startswith("t15.") or name.startswith("t12.")]
    return sorted(sessions)
