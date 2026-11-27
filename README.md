# FIPT: Spectral Cross-modal Prompting

Code for **Adapting Single-modal EEG Foundation Models to Multimodal Affective Computing via Spectral Cross-modal Prompting**.

FIPT adapts EEG foundation models to multimodal arousal and valence prediction. This repository provides LaBraM and CBraMod implementations for HCI, DEAP, MER, and REFED, with a separate model file and configuration for each dataset, backbone, and prediction target.

## Installation

Install a PyTorch build compatible with your CUDA environment, then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The training entry point runs on a single device. The default device is `cuda:0`.

## Pretrained weights

Place the backbone checkpoints at:

- LaBraM: `checkpoints/labram-base.pth`
- CBraMod: `checkpoints/pretrained_weights.pth`

## Data preparation

Follow the dataset preparation instructions in [preprocessing/README.md](preprocessing/README.md). Configure the dataset directories in `dataset_config/Classification.json`, or pass `--data-root` when running an experiment. The supplied configuration contains local experiment paths that should be updated for your environment.

Each preprocessed dataset directory must contain `train.json`, `val.json`, and `test.json`. Each manifest contains:

- `dataset_info`: dataset metadata, including `sampling_rate` and `ch_names`.
- `subject_data`: paths to preprocessed pickle files. Relative paths are resolved from the manifest directory.

Each pickle contains a `sample` array of shape `(channels, time)` and binary `label` values. The configurations expect data sampled at 200 Hz, with arousal at label index 1 and valence at label index 0. Channel order and split membership must match the dataset configuration. Input normalization is performed during preprocessing; the supplied training configurations disable additional loader normalization.

## Training

Run commands from the repository root. Select the dataset, backbone, and target explicitly:

```bash
python run_finetuning.py --dataset HCI --backbone LaBraM --target arousal
python run_finetuning.py --dataset DEAP --backbone CBraMod --target valence
```

Supported options are:

| Argument | Values |
| --- | --- |
| `--dataset` | `HCI`, `DEAP`, `MER`, `REFED` |
| `--backbone` | `LaBraM`, `CBraMod` |
| `--target` | `arousal`, `valence` |

Use `python run_finetuning.py --list` to inspect all 16 experiment configurations and `python run_finetuning.py --help` for available overrides. Without arguments, the current entry point runs **REFED / LaBraM / valence**.

Experiments use seeds **0, 1, 2, 3, 4** by default. Common settings are 50 epochs, batch size 128, AdamW, learning rate 1e-4, and weight decay 0.05. Dataset-specific settings are defined in `configs/<dataset>_<backbone>_<target>.json`. Both REFED–LaBraM configurations use binary label smoothing of 0.1 and a backbone learning-rate multiplier of 0.1. Other supplied configurations use unsmoothed binary cross-entropy and a multiplier of 1.0. Trainable parameters are determined by each model's freezing configuration.

Use `--seeds` to select individual runs and `--num_workers` to adjust data-loading workers (default: 20). Use a separate `--output_dir` for experiments with modified settings.

## Evaluation and results

Each run is saved under `finetuning_results/<dataset>_<backbone>_<target>/seed_<id>/`. The best checkpoint is selected by validation balanced accuracy (BA); ties retain the earliest epoch.

| File | Contents |
| --- | --- |
| `run_config.json` | Resolved configuration, seed, file hashes, and parameter counts |
| `log.txt` | Per-epoch training, validation, and test metrics |
| `checkpoint-best.pth`, `best.json` | Selected checkpoint and associated metrics |
| `checkpoint-last.pth` | Latest training state for resumption |
| `completed.json` | Completion record |

Summarize an experiment:

```bash
python read.py --results finetuning_results/hci_labram_arousal
```

The reader prints individual seed results and writes `summary.csv` and `summary.json`. It reports mean performance, population standard deviation (`ddof=0`), and sample standard deviation (`ddof=1`). Incomplete runs are excluded unless `--allow-partial` is supplied. Always specify `--results` for the experiment being analyzed; the reader's current default is `finetuning_results/mer_labram_valence`.

Evaluate a saved checkpoint:

```bash
python run_finetuning.py --dataset HCI --backbone LaBraM --target arousal --evaluate finetuning_results/hci_labram_arousal/seed_0/checkpoint-best.pth
```

Resume an interrupted run with its original training schedule and seed:

```bash
python run_finetuning.py --dataset HCI --backbone LaBraM --target arousal --seeds 0 --resume finetuning_results/hci_labram_arousal/seed_0/checkpoint-last.pth
```

## Reproducibility

Record the data splits, preprocessing, checkpoint versions, seeds, software versions, and device settings when reproducing an experiment. The dependency file specifies supported version ranges rather than a pinned experimental environment. Numerical results can vary across PyTorch, CUDA, and hardware versions. `--deterministic` enables deterministic PyTorch operations where supported.

HCI–CBraMod includes an initialization-time FLOPs trace whose forward pass consumes random numbers. Retain this initialization sequence when comparing runs with identical seeds.

Corresponding frozen backbone parameters with identical values share storage by default. Trainable parameters remain independent. Parameter reports distinguish logical parameter occurrences, unique stored parameters, and trainable parameters. `--no-share-frozen` disables storage sharing.

## Repository structure

- `models/`: complete model implementations for the 16 experiment configurations.
- `configs/`: training configurations and source provenance records.
- `preprocessing/`: dataset preparation scripts and documentation.
- `util/`: data loading, optimization, and parameter storage utilities.
- `run_finetuning.py`: training, evaluation, and resumption.
- `engine_for_finetuning.py`: training and evaluation loops.
- `read.py`: per-seed reporting and aggregate statistics.
