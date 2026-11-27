"""Dataset access for preprocessed AdaBrain-Bench classification manifests."""

import json
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """Load existing samples without resampling or applying normalization a second time.

    A manifest contains dataset_info and subject_data (a list of pickle paths).
    Each pickle contains sample: (channels, samples) and label: a sequence whose order is described by dataset_info.label_names.
    Legacy manifests without label_names retain the recipe label_index.
    Relative sample paths are resolved against the manifest's directory.
    """

    def __init__(self, manifest, label_index, channels, samples, target=None):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(encoding="utf-8-sig") as f:
            data = json.load(f)
        self.info = data["dataset_info"]
        self.files = data["subject_data"]
        self.label_index = label_index
        if target is not None and "label_names" in self.info:
            names = self.info["label_names"]
            if names.count(target) != 1:
                raise ValueError(f"{self.manifest}: expected one label named {target}")
            if names.index(target) != label_index:
                raise ValueError(f'{self.manifest}: label_names conflicts with the reference label_index={label_index}')
        self.shape = (channels, samples)
        self.ch_names = self.info["ch_names"]
        if self.info["sampling_rate"] != 200:
            raise ValueError(f"{self.manifest}: expected preprocessed 200 Hz data")
        eeg_channels = {36: 32, 20: 18, 166: 64}[channels]
        if len(self.ch_names) not in (eeg_channels, channels):
            raise ValueError(
                f"{self.manifest}: expected {eeg_channels} EEG names or {channels} total channel names"
            )
        if not self.files:
            raise ValueError(f"{self.manifest}: empty split")
        if not all(isinstance(p, str) for p in self.files):
            raise ValueError(
                "subject_data must contain paths to preprocessed pickle samples"
            )
        self.paths = [
            (Path(p) if Path(p).is_absolute() else self.manifest.parent / p).resolve()
            for p in self.files
        ]
        if len(self.paths) != len(set(self.paths)):
            raise ValueError(f"{self.manifest}: duplicate sample paths")
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples missing; first: {missing[0]}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with self.paths[index].open("rb") as f:
            sample = pickle.load(f)
        x = np.asarray(sample["sample"], dtype=np.float32)
        y = sample["label"][self.label_index]
        if x.shape != self.shape or not np.isfinite(x).all():
            raise ValueError(
                f"{self.paths[index]}: expected finite data of shape {self.shape}; got {x.shape}"
            )
        if y not in (0, 1):
            raise ValueError(f"{self.paths[index]}: binary label required, got {y!r}")
        return torch.from_numpy(x), torch.tensor(y)

    def get_ch_names(self):
        return list(self.ch_names)


def load_splits(root, recipe):
    datasets = [
        EEGDataset(
            Path(root) / (split + ".json"),
            recipe["label_index"],
            recipe["num_channels"],
            recipe["num_t"] * 200,
            target=recipe["target"],
        )
        for split in ("train", "val", "test")
    ]
    for i, left in enumerate(datasets):
        for right in datasets[i + 1 :]:
            if set(left.paths) & set(right.paths):
                raise ValueError(
                    "Sample paths overlap across train/validation/test splits"
                )
            if left.get_ch_names() != right.get_ch_names():
                raise ValueError("Channel order differs across data splits")
    return datasets
