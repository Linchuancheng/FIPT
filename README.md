# FIPT: Spectral Cross-modal Prompting

Implementation of **Adapting Single-modal EEG Foundation Models to Multimodal Affective Computing via Spectral Cross-modal Prompting**. FIPT adapts LaBraM and CBraMod to arousal and valence prediction on HCI, DEAP, MER and REFED.

## Organization

- `run_finetuning.py`: training, checkpoint evaluation and resumption on one device.
- `models/`: 16 complete dataset/backbone/target implementations. Each file contains its model components and classification head.
- `configs/`: experiment defaults, source identifiers and source-file SHA-256 hashes.
- `engine_for_finetuning.py`: training and evaluation.
- `util/`: data loading, optimization and frozen-weight storage sharing.
- `preprocessing/`: dataset preparation; see its README.
- `read.py`: per-seed results and aggregate statistics.

## Installation and data

Install a compatible PyTorch build, then:

```bash
python -m pip install -r requirements.txt
```

Place trusted pretrained weights in `checkpoints/labram-base.pth` and `checkpoints/pretrained_weights.pth`, or pass `--pretrained`.

`dataset_config/Classification.json` currently retains the experiment machine's paths. Update these paths or use `--data-root`. Each data directory must contain `train.json`, `val.json` and `test.json`. A manifest contains `dataset_info` (including `sampling_rate` and `ch_names`) and `subject_data`, a list of pickle paths. Each pickle contains `sample` with shape `(channels, time)` and binary `label` values. Relative paths resolve from the manifest directory.

Use the same preprocessed files and split membership as the reference experiment. The supplied recipes use 200 Hz data and disable additional loader normalization. The loader checks dimensions, finite values and split path overlap. The recipe's explicit label index determines the target; conflicting `label_names` metadata raises an error instead of silently changing the target. The current reference files use label index 1 for arousal and 0 for valence.

## Training

```bash
python run_finetuning.py --dataset HCI --backbone LaBraM --target arousal
python run_finetuning.py --dataset DEAP --backbone CBraMod --target valence
python run_finetuning.py --list
```

Running without options uses **MER / LaBraM / valence**, matching the current entry-point defaults. To select another experiment, use `--dataset {HCI,DEAP,MER,REFED}`, `--backbone {LaBraM,CBraMod}` and `--target {arousal,valence}`. `--model_name` is an alias for `--backbone`.

All recipes default to seeds **0, 1, 2, 3, 4**. Common settings are 50 epochs, batch size 128, AdamW, learning rate 1e-4 and weight decay 0.05. The JSON recipe is the authoritative configuration. Both REFED–LaBraM targets use binary label smoothing 0.1 and a 0.1 learning-rate multiplier for eligible existing-backbone parameters; the remaining supplied recipes use unsmoothed BCE and multiplier 1.0. The historical `finetune_mod=full` name retains the recipe's selective freezing mask; it does not mean that every parameter is trainable.

Useful options include `--data-root`, `--pretrained`, `--device cuda:0`, `--num_workers`, `--seeds`, `--lr` and `--output_dir`. The worker default is 20; lower it for machines with fewer resources. `--from-scratch` is a diagnostic mode and does not reproduce pretrained experiments.

## Outputs and model selection

Results are written to `finetuning_results/<recipe>/seed_<id>/`:

- `run_config.json`: resolved settings, seed, source mapping, recipe/model hashes and parameter counts.
- `log.txt`: training, validation and test metrics for each epoch.
- `checkpoint-best.pth` and `best.json`: highest validation balanced accuracy, with the earliest epoch retained on ties.
- `checkpoint-last.pth`: model, optimizer, scaler and random states for resumption.
- `completed.json`: completion marker.

Test results do not select the checkpoint. Existing logs are protected from accidental appending; use a new output directory for a new experiment.

```bash
python read.py --results finetuning_results/hci_labram_arousal
python run_finetuning.py --dataset HCI --backbone LaBraM --target arousal --seeds 0 --resume finetuning_results/hci_labram_arousal/seed_0/checkpoint-last.pth
python run_finetuning.py --dataset HCI --backbone LaBraM --target arousal --evaluate finetuning_results/hci_labram_arousal/seed_0/checkpoint-best.pth
```

The reader prints each seed and writes `summary.csv` and `summary.json`. It reports both population SD (`ddof=0`) and sample SD (`ddof=1`), and excludes incomplete runs unless `--allow-partial` is specified. With no options it reads the default MER / LaBraM / valence experiment. Resume requires the original schedule and exactly one seed.

## Reference compatibility

The configurations and complete model files are synchronized with the supplied `best` experiments. Two compatibility details are explicit:

1. HCI–LaBraM arousal returns a three-dimensional PSD tensor. Its source caller contained an extra flatten operation that merged tokens with frequency bins and prevented execution. The consolidated caller omits that operation and retains the source PSD formula.
2. Both this repository and the standalone best scripts save checkpoints by the highest validation BA, retaining the earliest epoch on ties. Compare runs using the same seed IDs.

HCI–CBraMod retains the source initialization-time FLOPs trace: its random input generation and forward execution affect initialization/RNG state. Removing that trace changes seed-by-seed reproduction.

Local verification used the supplied pretrained weights and synthetic inputs on CPU. All 16 recipes matched source parameter initialization, trainability masks, evaluation/training outputs, gradients and weights after two optimizer steps. HCI–LaBraM arousal was compared after applying the dimensional correction described above. Dataset-loading, optimizer-group and binary-loss checks also passed for all 16 recipes. These are implementation-equivalence checks, not full-dataset benchmark results.

For numerical reproduction, use the same data, pretrained weights, seeds, PyTorch/timm/CUDA versions and device settings. `--deterministic` requests deterministic PyTorch operations; it differs from the original cuDNN benchmark setting and does not guarantee agreement across environments.

## Frozen-weight storage

Only corresponding frozen parameters with exactly equal values share storage. Trainable parameters remain independent. This preserves modality computations and dropout calls while reducing duplicate storage. Use `--no-share-frozen` for an unshared comparison. Checkpoints retain their state-dictionary keys; loading separates aliases before restoring weights and shares only tensors that remain equal.

Parameter reports distinguish logical parameter occurrences, unique stored parameters and trainable parameters. State which denominator is used when reporting parameter ratios.

## Attribution

See `THIRD_PARTY_NOTICES.md` and `licenses/` for AdaBrain-Bench, LaBraM, CBraMod and dependency attribution. Cite the FIPT paper and the backbone/framework papers used in your experiments. Dataset files and pretrained weights are distributed separately.
