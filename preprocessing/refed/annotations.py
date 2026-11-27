"""REFED preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
import os
from scipy.io import loadmat
import gc


def load_data(data_path, sub_list=None, modality: list = ["EEG", "fNIRS"]):
    """Read dataset data using the supplied source layout."""
    data = {}
    if sub_list is None:
        sub_list = os.listdir(data_path)
    for si in sub_list:
        if os.path.isdir(os.path.join(data_path, si)):
            data[si] = {}
            if "EEG" in modality:
                path_si_EEG = os.path.join(data_path, si, "EEG_videos.mat")
                data_si_EEG = loadmat(path_si_EEG)
                data[si]["EEG"] = {
                    "v%d" % vi: data_si_EEG["video_%d" % vi] for vi in range(1, 16)
                }
                del data_si_EEG
            if "fNIRS" in modality:
                path_si_fNIRS = os.path.join(data_path, si, "fNIRS_videos.mat")
                data_si_fNIRS = loadmat(path_si_fNIRS)
                data[si]["fNIRS"] = {
                    "v%d" % vi: data_si_fNIRS["video_%d" % vi] for vi in range(1, 16)
                }
                del data_si_fNIRS
            gc.collect()
    return data


def load_baseline(data_path, sub_list=None, modality: list = ["EEG", "fNIRS"]):
    """Read baseline recordings from the original MATLAB layout."""
    data = {}
    if sub_list is None:
        sub_list = os.listdir(data_path)
    for si in sub_list:
        if os.path.isdir(os.path.join(data_path, si)):
            data[si] = {}
            if "EEG" in modality:
                path_si_EEG = os.path.join(data_path, si, "EEG_baselines.mat")
                data_si_EEG = loadmat(path_si_EEG)
                data[si]["EEG"] = {
                    "v%d" % vi: data_si_EEG["video_%d" % vi] for vi in range(1, 16)
                }
                del data_si_EEG
            if "fNIRS" in modality:
                path_si_fNIRS = os.path.join(data_path, si, "fNIRS_baselines.mat")
                data_si_fNIRS = loadmat(path_si_fNIRS)
                data[si]["fNIRS"] = {
                    "v%d" % vi: data_si_fNIRS["video_%d" % vi] for vi in range(1, 16)
                }
                del data_si_fNIRS
            gc.collect()
    return data


def load_feature(data_path, sub_list=None, modality: list = ["EEG", "fNIRS"]):
    """Read precomputed features from the original MATLAB layout."""
    data = {}
    if sub_list is None:
        sub_list = os.listdir(data_path)
    for si in sub_list:
        if os.path.isdir(os.path.join(data_path, si)):
            data[si] = {}
            if "EEG" in modality:
                path_si_EEG = os.path.join(data_path, si, "EEG_videos_feature.mat")
                data_si_EEG = loadmat(path_si_EEG)
                data[si]["EEG"] = {
                    "v%d" % vi: data_si_EEG["video_%d" % vi] for vi in range(1, 16)
                }
                del data_si_EEG
            if "fNIRS" in modality:
                path_si_fNIRS = os.path.join(data_path, si, "fNIRS_videos_feature.mat")
                data_si_fNIRS = loadmat(path_si_fNIRS)
                data[si]["fNIRS"] = {
                    "v%d" % vi: data_si_fNIRS["video_%d" % vi] for vi in range(1, 16)
                }
                del data_si_fNIRS
            gc.collect()
    return data


def load_label(data_path, sub_list=None, dimension: list = ["Valence", "Arousal"]):
    """Read dynamic annotations from the original MATLAB layout."""
    label = {}
    if sub_list is None:
        sub_file = os.listdir(data_path)
    else:
        sub_file = ["s%s_label.mat" % si for si in sub_list]
    for si in sub_file:
        if si.endswith("_label.mat"):
            si_key = "s%s" % si[:-10]
            file_path = os.path.join(data_path, si)
            label_si = loadmat(file_path)
            label[si_key] = {}
            for vi in range(1, 16):
                vi_key = "v%d" % vi
                label[si_key][vi_key] = {}
                if "Valence" in dimension:
                    label[si_key][vi_key]["Valence"] = label_si["video_%d" % vi][:, 0]
                if "Arousal" in dimension:
                    label[si_key][vi_key]["Arousal"] = label_si["video_%d" % vi][:, 1]
    return label


def run(settings):
    """Execute this stage using the original processing order."""
    config = settings
