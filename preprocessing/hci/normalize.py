"""HCI preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
import pickle
import numpy as np
import json


def normalize(X, mean_value, std_value):
    """Apply the original training-derived channel normalization."""
    mean_value, std_value = (np.array(mean_value), np.array(std_value))
    mu, sigma = (np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1))
    X = (X - mu) / (sigma + 1e-08)
    return X


def compute_normalization_params(mode):
    """Average training-window means and standard deviations per channel."""
    total_mean = np.zeros(len(ch))
    total_std = np.zeros(len(ch))
    total = 0
    for sub_id in trials_split[mode]:
        files_name = loaded_data[sub_id]
        for file in files_name:
            with open(file, "rb") as f:
                data = pickle.load(f)
            sample = data["sample"]
            for j in range(len(ch)):
                total_mean[j] += sample[j].mean()
                total_std[j] += sample[j].std()
            total += 1
    mean = (total_mean / total).tolist()
    std = (total_std / total).tolist()
    return (mean, std)


def run(settings):
    """Execute this stage using the original processing order."""
    global ch, loaded_data, trials_split
    config = settings
    file_path = config.output + "/data/all_data.pkl"
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    keys = [i for i in list(loaded_data.keys())]
    keys.sort()
    trials_split = {"train": keys[:19], "val": keys[19:23], "test": keys[23:27]}
    dataset = {"train": list(), "val": list(), "test": list()}
    ch = [
        "Fp1",
        "AF3",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "P3",
        "P7",
        "PO3",
        "O1",
        "Oz",
        "Pz",
        "Fp2",
        "AF4",
        "Fz",
        "F4",
        "F8",
        "FC6",
        "FC2",
        "Cz",
        "C4",
        "T8",
        "CP6",
        "CP2",
        "P4",
        "P8",
        "PO4",
        "O2",
        "EXG1",
        "EXG2",
        "EXG3",
        "GSR1",
    ]
    mean, std = compute_normalization_params("train")
    for mode in trials_split.keys():
        p_files = []
        print(mode)
        for sub_id in trials_split[mode]:
            print(sub_id)
            files_name = loaded_data[sub_id]
            for file in files_name:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                data["sample"] = normalize(data["sample"], mean, std)
                with open(f"{file}", "wb") as f:
                    pickle.dump(data, f)
                p_files.append(file)
        data_info = {
            "subject_data": p_files,
            "dataset_info": {
                "sampling_rate": 200,
                "ch_names": ch,
                "min": 0,
                "max": 0,
                "mean": mean,
                "std": std,
            },
        }
        with open(f"{config.output}/{mode}.json", "w") as f:
            json.dump(data_info, f, indent=2)
