"""REFED preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
from scipy.io import loadmat
import os
import numpy as np
from preprocessing.refed.annotations import load_label
import pandas as pd
import pickle
import json


def run(settings):
    """Execute this stage using the original processing order."""
    config = settings
    path_mat = config.intermediate + "/"
    sub_list = [s for s in os.listdir(path_mat) if s.isdigit()]
    sub_list.sort()
    keys = [f"video_{i}" for i in range(1, 16)]
    with open(str(Path(__file__).with_name("ch_info_EEG.pkl")), "rb") as f:
        eeg_ch = pickle.load(f)
    with open(str(Path(__file__).with_name("ch_info_fNIRS.pkl")), "rb") as f:
        fNIRS_ch = pickle.load(f)
    trials_split = {"train": range(1, 25), "val": range(25, 29), "test": range(29, 33)}
    dataset = {"train": list(), "val": list(), "test": list()}
    path_annotation = config.annotations
    path_SAM = config.sam
    SAM_score = pd.read_csv(path_SAM)
    label_all = load_label(path_annotation)
    Dynamic_Valence = [
        np.array([label_all["s%d" % si]["v%d" % vi]["Valence"] for si in range(1, 33)])
        / 256
        for vi in range(1, 16)
    ]
    Dynamic_Arousal = [
        np.array([label_all["s%d" % si]["v%d" % vi]["Arousal"] for si in range(1, 33)])
        / 256
        for vi in range(1, 16)
    ]
    for mode in dataset.keys():
        for sub in trials_split[mode]:
            sub = str(sub)
            print("sub: ", sub)
            eeg_fnirs_file = os.listdir(path_mat + sub)
            sub_eeg_data = loadmat(
                path_mat + sub + "/" + [i for i in eeg_fnirs_file if "EEG" in i][0]
            )
            sub_fnirs_data = loadmat(
                path_mat + sub + "/" + [i for i in eeg_fnirs_file if "fNIRS" in i][0]
            )
            labels = SAM_score[SAM_score["sub_id"] == int(sub)]
            for video_id in range(1, 16):
                print(video_id)
                video_eeg_data = sub_eeg_data[f"video_{video_id}"]
                video_fnirs_data = sub_fnirs_data[f"video_{video_id}"][:, :2, :, :]
                video_fnirs_data = video_fnirs_data.reshape(
                    video_fnirs_data.shape[0], -1, video_fnirs_data.shape[-1]
                )
                con_data = np.hstack([video_eeg_data, video_fnirs_data])
                for segment in range(con_data.shape[0]):
                    label = [
                        labels[f"Video_{video_id}_Arousal"].tolist()[0] // 5,
                        labels[f"Video_{video_id}_Valence"].tolist()[0] // 5,
                        Dynamic_Valence[video_id - 1][int(sub) - 1][segment],
                        Dynamic_Arousal[video_id - 1][int(sub) - 1][segment],
                    ]
                    data_dict = {"sample": con_data[segment], "label": label}
                    sample_key = f"{config.output + '/data/'}{sub}/{sub}-{video_id}-{segment}.pkl"
                    os.makedirs(f"{config.output + '/data/'}{sub}", exist_ok=True)
                    with open(f"{sample_key}", "wb") as f:
                        pickle.dump(data_dict, f)
                    dataset[mode].append(sample_key)
    for mode in dataset.keys():
        new_dataset = {
            "subject_data": dataset[mode],
            "dataset_info": {
                "sampling_rate": 200,
                "ch_names": eeg_ch[1],
                "min": 0,
                "max": 0,
                "mean": 0,
                "std": 0,
            },
        }
        with open(f"{config.output}/{mode}.json", "w") as f:
            json.dump(new_dataset, f, indent=2)
    print("success")
