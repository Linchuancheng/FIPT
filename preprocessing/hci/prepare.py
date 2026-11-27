"""HCI preprocessing stage; signal operations follow the supplied implementation."""

from pathlib import Path
import glob
import json
import os
from typing import Any, Callable, Dict, Tuple, Union
import mne
import xmltodict
import numpy as np
from scipy.signal import filtfilt, firwin
import pickle


def bandpass_filter(data, lowcut=4.0, highcut=45.0, fs=128, numtaps=101):
    """Apply the original FIR bandpass with zero-phase filtering."""
    nyq = fs / 2
    taps = firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=False)
    y = filtfilt(taps, 1.0, data)
    return y


def resample_signal(raw_segment, target_sfreq):
    """Resample the actual signal and verify its rate and sample count."""
    expected_samples = int(
        round(raw_segment.n_times * target_sfreq / raw_segment.info["sfreq"])
    )
    raw_segment.resample(target_sfreq)
    if not np.isclose(raw_segment.info["sfreq"], target_sfreq):
        raise RuntimeError("Unexpected sampling rate after resampling")
    signal = raw_segment.get_data()
    if signal.shape[1] != expected_samples:
        raise RuntimeError("Unexpected sample count after resampling")
    return signal


def run(settings):
    """Execute this stage using the original processing order."""
    config = settings
    target_sampling_rate = 200
    chunk_size = target_sampling_rate * 4
    overlap = 0
    num_baseline = 5
    baseline_chunk_size = target_sampling_rate * 6
    num_channel = 36
    num_trial_sample = -1
    root_path = config.source
    files = os.listdir(root_path)
    all_sub = [
        "21",
        "6",
        "30",
        "17",
        "22",
        "7",
        "9",
        "14",
        "28",
        "4",
        "18",
        "16",
        "24",
        "20",
        "19",
        "10",
        "13",
        "5",
        "29",
        "27",
        "23",
        "3",
        "11",
        "25",
        "1",
        "8",
        "2",
    ]
    all_data = {int(sub_id): [] for sub_id in all_sub}
    for file_name in files:
        print(file_name)
        trial_dir = os.path.join(root_path, file_name)
        mne.set_log_level("CRITICAL")
        label_file = os.path.join(trial_dir, "session.xml")
        emodims = [
            "@feltArsl",
            "@feltCtrl",
            "@feltEmo",
            "@feltPred",
            "@feltVlnc",
            "@isStim",
        ]
        with open(label_file) as f:
            label_info = xmltodict.parse("\n".join(f.readlines()))
        label_info = json.loads(json.dumps(label_info))["session"]
        if not "@feltArsl" in label_info:
            continue
        trial_meta_info = {
            "subject_id": label_info["subject"]["@id"],
            "trial_id": label_info["@mediaFile"],
            "duration": float(label_info["@cutLenSec"]),
        }
        trial_meta_info.update({k[1:]: int(label_info[k]) for k in emodims})
        write_pointer = 0
        sample_file = glob.glob(str(os.path.join(trial_dir, "*.bdf")))[0]
        raw = mne.io.read_raw_bdf(sample_file, preload=True, stim_channel="Status")
        raw_sampling_rate = float(raw.info["sfreq"])
        print("raw_sampling_rate", raw_sampling_rate)
        source_first_samp = raw.first_samp
        events = mne.find_events(raw, stim_channel="Status")
        montage = mne.channels.make_standard_montage(kind="biosemi32")
        raw.set_montage(montage, on_missing="ignore")
        need_ch = [
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
        raw.pick_channels(need_ch, ordered=True)
        tmin, tmax = (raw.times[0], raw.times[-1])
        new_tmax = tmax - 2
        raw.crop(tmin=tmin, tmax=new_tmax)
        raw_info = raw.info.copy()
        signal = raw.get_data()
        filtered_signal = np.zeros_like(signal, dtype=np.float64)
        for i in range(32):
            filtered_signal[i] = bandpass_filter(signal[i], 0.1, 80, raw_sampling_rate)
        for i in range(32, 35):
            filtered_signal[i] = bandpass_filter(signal[i], 0.05, 95, raw_sampling_rate)
        for i in range(35, 36):
            filtered_signal[i] = bandpass_filter(signal[i], 0.05, 5, raw_sampling_rate)
        raw = mne.io.RawArray(filtered_signal, raw_info)
        start_samp = int(events[0][0] - source_first_samp) + 1
        end_samp = int(events[1][0] - source_first_samp) - 1
        if not 0 <= start_samp < end_samp < raw.n_times:
            raise ValueError(
                f"Invalid event interval in session {file_name}: {start_samp}:{end_samp}"
            )
        trial_baseline_raw = raw.copy().crop(raw.times[0], raw.times[start_samp])
        trial_baseline_sample = resample_signal(
            trial_baseline_raw, target_sampling_rate
        )
        trial_baseline_sample = trial_baseline_sample[
            :, : num_baseline * baseline_chunk_size
        ]
        trial_baseline_sample = trial_baseline_sample.reshape(
            num_channel, num_baseline, baseline_chunk_size
        ).mean(axis=1)
        trial_raw = raw.copy().crop(raw.times[start_samp], raw.times[end_samp])
        trial_samples = resample_signal(trial_raw, target_sampling_rate)
        print(
            f"Session {file_name}: {raw_sampling_rate:g} -> {target_sampling_rate:g} Hz; trial shape={trial_samples.shape}; window={chunk_size} samples (4 s)"
        )
        start_at = 0
        if chunk_size <= 0:
            dynamic_chunk_size = trial_samples.shape[1] - start_at
        else:
            dynamic_chunk_size = chunk_size
        end_at = dynamic_chunk_size
        step = dynamic_chunk_size - overlap
        max_len = trial_samples.shape[1]
        if not num_trial_sample <= 0:
            max_len = min(num_trial_sample * dynamic_chunk_size, trial_samples.shape[1])
        while end_at <= max_len:
            clip_sample = trial_samples[:, start_at:end_at]
            t_eeg = clip_sample
            t_baseline = trial_baseline_sample
            if not "baseline_id" in trial_meta_info:
                trial_base_id = f"{file_name}_{write_pointer}"
                data = {"eeg": t_baseline, "key": trial_base_id}
                write_pointer += 1
                trial_meta_info["baseline_id"] = trial_base_id
            clip_id = f"{file_name}_{write_pointer}"
            write_pointer += 1
            record_info = {"start_at": start_at, "end_at": end_at, "clip_id": clip_id}
            record_info.update(trial_meta_info)
            data = {"eeg": t_eeg, "key": clip_id, "info": record_info}
            data_dict = {
                "sample": t_eeg,
                "label": [
                    int(record_info["feltVlnc"] / 5),
                    int(record_info["feltArsl"] / 5),
                ],
            }
            sub_id = label_info["subject"]["@id"]
            trial_id = label_info["@mediaFile"].replace(".avi", "")
            os.makedirs(f"{config.output + '/data' + '/'}{sub_id}", exist_ok=True)
            sample_key = f"{config.output + '/data' + '/'}{sub_id}/{sub_id}-{trial_id}-{start_at}.pkl"
            with open(f"{sample_key}", "wb") as f:
                pickle.dump(data_dict, f)
            all_data[int(sub_id)].append(sample_key)
            start_at = start_at + step
            end_at = start_at + dynamic_chunk_size
    with open(f"{config.output + '/data' + '/all_data.pkl'}", "wb") as f:
        pickle.dump(all_data, f)
    print("success")
