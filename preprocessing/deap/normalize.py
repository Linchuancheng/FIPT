"""DEAP preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
import json
import pickle
import numpy as np


def normalize(X, mean_value, std_value):
    """Apply the original training-derived channel normalization."""
    mean_value, std_value = (np.array(mean_value), np.array(std_value))
    mu, sigma = (np.expand_dims(mean_value, axis=1), np.expand_dims(std_value, axis=1))
    X = (X - mu) / (sigma + 1e-08)
    return X


def run(settings):
    """Execute this stage using the original processing order."""
    config = settings
    num_channels = 36
    total_mean = np.zeros(num_channels)
    total_std = np.zeros(num_channels)
    with open(f"{config.output + '/train.json'}", "rb") as f:
        files = json.load(f)
    total = 0
    for file in files["subject_data"]:
        with open(file, "rb") as f:
            data = pickle.load(f)
        for j in range(num_channels):
            total_mean[j] += data["sample"][j].mean()
            total_std[j] += data["sample"][j].std()
        total += 1
    mean = (total_mean / total).tolist()
    std = (total_std / total).tolist()
    for mode in ["train", "val", "test"]:
        with open(f"{config.output + '/'}{mode}.json", "rb") as f:
            files = json.load(f)
        for file in files["subject_data"]:
            with open(file, "rb") as f:
                data = pickle.load(f)
            data["sample"] = normalize(data["sample"], mean, std)
            with open(f"{file}", "wb") as f:
                pickle.dump(data, f)
