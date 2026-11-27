"""DEAP preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
import os
import numpy
import h5py
import scipy.io as sio
import re
import numpy as np
from scipy import signal
import pandas as pd
from scipy.signal import resample
import pickle
import json
import mne.filter


def get_p_labels(mat_labels):
    """Binarize valence and arousal ratings using the original threshold."""
    labels = []
    for i in mat_labels:
        p = []
        if i[0] >= 5:
            p.append(1)
        else:
            p.append(0)
        if i[1] >= 5:
            p.append(1)
        else:
            p.append(0)
        labels.append(p)
    return labels


def data_resample(data):
    """Filter/resample signals using the dataset-specific original settings."""
    n_channels, n_old_points = data.shape
    f_old = 128
    f_new = 200
    n_new_points = int(np.ceil(f_new / f_old * n_old_points))
    resampled_trial_data = resample(data, n_new_points, axis=-1)
    return resampled_trial_data


def load_data(timeLen, timeStep):
    """Read dataset data using the supplied source layout."""
    n_samples_old = [60] * 40
    n_samples_remain = np.zeros(len(n_samples_old), dtype=int)
    n_samples = np.zeros(len(n_samples_old), dtype=int)
    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples_old[i] - timeLen) / timeStep + 1)
        n_samples_remain[i] = n_samples_old[i] - n_samples[i] * timeStep
    return (n_samples, n_samples_remain)


def prepare(x: np.ndarray, sfreq, l_freq, h_freq) -> np.ndarray:
    """Apply the original modality-specific temporal filters."""
    x = mne.filter.filter_data(
        data=x,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        fir_design="firwin",
        verbose=False,
    )
    notch_freq = 50.0
    x = mne.filter.notch_filter(
        x, Fs=sfreq, freqs=notch_freq, method="fir", verbose=False
    )
    return x


def run(settings):
    """Execute this stage using the original processing order."""
    config = settings
    ch = ["Fp1","AF3","F3","F7","FC5","FC1","C3","T7","CP5","CP1","P3","P7","PO3","O1","Oz","Pz","Fp2","AF4","Fz","F4","F8","FC6","FC2","Cz","C4","T8","CP6","CP2","P4","P8","PO4","O2","hEOG","vEOG","zEMG","tEMG",]
    num_channels = len(ch)
    files_name = os.listdir(config.source)
    df = pd.read_csv(config.ratings)
    ex_id = df["Experiment_id"]
    number_ex_id = []
    trials_split = {"train": range(24), "val": range(24, 28), "test": range(28, 32)}
    dataset = {"train": list(), "val": list(), "test": list()}
    for mode in trials_split.keys():
        print(mode)
        print("success step 1")
        timeLen = 4
        timeStep = 4
        for file in trials_split[mode]:
            file_name = [i for i in files_name if int(i[1:3]) == file + 1][0]
            print(file_name)
            mat = sio.loadmat(config.source + "/" + file_name)
            data = mat["data"][:, :-4, 384:]
            p_labels = get_p_labels(mat["labels"])
            n_samples, n_samples_remain = load_data(timeLen, timeStep)
            for index in range(40):
                trial_data = data[index]
                label = p_labels[index]
                trial_data_eeg = prepare(trial_data[:32, :], 128, 0.05, 60)
                trial_data_eog = prepare(trial_data[32:34, :], 128, 0.05, 50)
                trial_data_emg = prepare(trial_data[34:36, :], 128, 10, 60)
                trial_data = np.concatenate(
                    [trial_data_eeg, trial_data_eog, trial_data_emg], axis=0
                )
                trial_data = data_resample(trial_data)
                for sample_num in range(n_samples[index]):
                    sample = trial_data[
                        :,
                        sample_num
                        * timeStep
                        * 200 : (sample_num * timeStep + timeLen)
                        * 200,
                    ]
                    sample_key = f"{config.output + '/data/'}{file}/{file}-{index}-{sample_num}.pkl"
                    data_dict = {"sample": sample, "label": label}
                    os.makedirs(f"{config.output + '/data/'}{file}", exist_ok=True)
                    with open(f"{sample_key}", "wb") as f:
                        pickle.dump(data_dict, f)
                    dataset[mode].append(sample_key)
            new_dataset = {
                "subject_data": dataset[mode],
                "dataset_info": {
                    "sampling_rate": 200,
                    "ch_names": ch,
                    "min": 0,
                    "max": 0,
                    "mean": 0,
                    "std": 0,
                },
            }
            with open(f"{config.output}/{mode}.json", "w") as f:
                json.dump(new_dataset, f, indent=2)
