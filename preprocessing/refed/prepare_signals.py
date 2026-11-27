"""REFED preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
import os
import mne
from scipy.io import loadmat, savemat


def preprocess_EEG_sub(path_data, para_preprocess, save_path=None):
    """Resample and window EEG recordings for one subject."""
    EEG_data = loadmat(os.path.join(path_data, "EEG_videos.mat"))
    EEG_preprocessed = {}
    for vi in EEG_data.keys():
        if "video_" in vi:
            EEG_vi = mne.filter.resample(
                EEG_data[vi],
                up=para_preprocess["EEG_fs_down"],
                down=para_preprocess["EEG_fs"],
                verbose=False,
            )
            points_per_window = (
                para_preprocess["window"] * para_preprocess["EEG_fs_down"]
            )
            remainder = EEG_vi.shape[-1] % points_per_window
            if remainder != 0:
                print("sub:", path_data)
                print(remainder)
                EEG_vi = EEG_vi[:, :-remainder]
                print(
                    f"  [Warning] Video {vi}: Truncated {remainder} points to fit {para_preprocess['window']}s window."
                )
            EEG_vi = EEG_vi.reshape(
                EEG_vi.shape[0],
                -1,
                para_preprocess["window"] * para_preprocess["EEG_fs_down"],
            ).transpose([1, 0, 2])
            print(
                "  [info] Video %s: %s -> %s" % (vi, EEG_data[vi].shape, EEG_vi.shape)
            )
            EEG_preprocessed[vi] = EEG_vi
    if save_path is not None:
        preprocessed_mat_path = os.path.join(save_path, "EEG_videos.mat")
        savemat(preprocessed_mat_path, EEG_preprocessed)
        print("[info] Save preprocessed data to: %s" % preprocessed_mat_path)


def preprocess_fNIRS_sub(path_data, para_preprocess, save_path=None):
    """Filter, resample and window fNIRS recordings for one subject."""
    fNIRS_data = loadmat(os.path.join(path_data, "fNIRS_videos.mat"))
    fNIRS_preprocessed = {}
    for vi in fNIRS_data.keys():
        if "video_" in vi:
            fNIRS_vi = mne.filter.filter_data(
                fNIRS_data[vi],
                sfreq=para_preprocess["fNIRS_fs"],
                l_freq=para_preprocess["fNIRS_filt_band"][0],
                h_freq=para_preprocess["fNIRS_filt_band"][1],
                method="iir",
                verbose=False,
            )
            fNIRS_vi = mne.filter.resample(
                fNIRS_vi, up=25, down=para_preprocess["fNIRS_fs"], verbose=False
            )
            fNIRS_vi = mne.filter.resample(
                fNIRS_vi, up=para_preprocess["fNIRS_fs_down"], down=25, verbose=False
            )
            points_per_window = (
                para_preprocess["window"] * para_preprocess["fNIRS_fs_down"]
            )
            remainder = fNIRS_vi.shape[-1] % points_per_window
            if remainder != 0:
                print("sub:", path_data)
                print(remainder)
                fNIRS_vi = fNIRS_vi[:, :, :-remainder]
                print(
                    f"  [Fixed] Video {vi}: Truncated {remainder} points to align with {para_preprocess['window']}s window."
                )
            if fNIRS_vi.shape[-1] % para_preprocess["fNIRS_fs_down"] != 0:
                if fNIRS_vi.shape[-1] % para_preprocess["fNIRS_fs_down"] == 1:
                    fNIRS_vi = fNIRS_vi[:, :, :-1]
                else:
                    raise NotImplementedError(
                        "  [error] Video %s: fNIRS data length cannot be divided by %d after downsampling!"
                        % (vi, para_preprocess["fNIRS_fs_down"])
                    )
            points_per_window = (
                para_preprocess["window"] * para_preprocess["fNIRS_fs_down"]
            )
            current_len = fNIRS_vi.shape[-1]
            remainder = current_len % points_per_window
            if remainder != 0:
                fNIRS_vi = fNIRS_vi[:, :, :-remainder]
                print(
                    f"  [Warning] Video {vi}: fNIRS truncated {remainder} points to fit {para_preprocess['window']}s window."
                )
            fNIRS_vi = fNIRS_vi.reshape(
                fNIRS_vi.shape[0],
                fNIRS_vi.shape[1],
                -1,
                para_preprocess["window"] * para_preprocess["fNIRS_fs_down"],
            ).transpose([2, 0, 1, 3])
            print(
                "  [info] Video %s: %s -> %s"
                % (vi, fNIRS_data[vi].shape, fNIRS_vi.shape)
            )
            fNIRS_preprocessed[vi] = fNIRS_vi
    if save_path is not None:
        preprocessed_mat_path = os.path.join(save_path, "fNIRS_videos.mat")
        savemat(preprocessed_mat_path, fNIRS_preprocessed)
        print("[info] Save preprocessed data to: %s" % preprocessed_mat_path)


def run(settings):
    """Execute this stage using the original processing order."""
    config = settings
    path_data = config.source
    path_preprocessed = config.intermediate
    para_preprocess = {
        "EEG_fs": 1000,
        "EEG_fs_down": 200,
        "EEG_filt_band": [0.5, 80],
        "EEG_filt_north": [50],
        "fNIRS_fs": 1000 / 21,
        "fNIRS_fs_down": 200,
        "fNIRS_filt_band": [0.01, 10],
        "window": 1,
    }
    sub_list = [s for s in os.listdir(path_data) if s.isdigit()]
    if not os.path.exists(path_preprocessed):
        os.makedirs(path_preprocessed)
    for si in sub_list:
        print("Processing EEG data of subject ID.[%s]..." % si)
        path_data_si = os.path.join(path_data, si)
        path_preprocessed_si = os.path.join(path_preprocessed, si)
        if not os.path.exists(path_preprocessed_si):
            os.makedirs(path_preprocessed_si)
        preprocess_EEG_sub(
            path_data_si, para_preprocess, save_path=path_preprocessed_si
        )
    for si in sub_list:
        print("Processing fNIRS data of subject ID.[%s]..." % si)
        path_data_si = os.path.join(path_data, si)
        path_preprocessed_si = os.path.join(path_preprocessed, si)
        if not os.path.exists(path_preprocessed_si):
            os.makedirs(path_preprocessed_si)
        preprocess_fNIRS_sub(
            path_data_si, para_preprocess, save_path=path_preprocessed_si
        )
