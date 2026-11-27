# Dataset preprocessing

These scripts package the supplied DEAP, HCI, MER and REFED pipelines. Signal
operations, channel ordering, label thresholds, sample enumeration and split
rules are retained. Paths are configurable and each pipeline runs through one
entry point. Raw recordings and annotations are distributed by the dataset owners.

From the **FIPT repository root**, install the additional dependencies and run:

```bash
python -m pip install -r preprocessing/requirements.txt

python -m preprocessing.run --dataset DEAP --source /data/DEAP/data_preprocessed_matlab --ratings /data/DEAP/participant_ratings.csv --output /data/FIPT_ready/DEAP
python -m preprocessing.run --dataset HCI --source /data/HCI/Sessions/Sessions/Sessions --output /data/FIPT_ready/HCI
python -m preprocessing.run --dataset MER --source /data/MER/Aligned_data_001 --labels /data/MER/Raw_data --output /data/FIPT_ready/MER
python -m preprocessing.run --dataset REFED --source /data/REFED --output /data/FIPT_ready/REFED
```

Replace the example paths with your actual locations. On Windows, quote paths
containing spaces. The output directory must be new or empty and outside the raw
dataset tree. This prevents accidental in-place normalization of existing outputs.
An interrupted pipeline leaves a `preprocessing_run.json` record; rerun into a new
output directory. There is no automatic resume through partially normalized files.

## Stages

| Dataset | Execution order | Preserved sample and split definition |
|---|---|---|
| DEAP | `deap/prepare.py` → `deap/normalize.py` | 36 channels; remove the initial 384 points at 128 Hz; resample to 200 Hz; 4 s windows, 4 s stride; subjects 1–24 / 25–28 / 29–32 |
| HCI | `hci/prepare.py` → `hci/normalize.py` | 36 channels; crop using original event indices, then resample to 200 Hz; 4 s windows, no overlap; sorted available subject IDs split 19 / 4 / 4 |
| MER | `mer/prepare.py` → `mer/normalize.py` | 18 EEG + GSR + PPG; resample to 200 Hz; 4 s windows with 1 s overlap; **stimulus IDs**, not subject IDs: 1–23 / 24–27 / 28–31 |
| REFED | `refed/prepare_signals.py` → `refed/build_samples.py` → `refed/normalize.py` | EEG + first two fNIRS components; 166 channels, 1 s windows at 200 Hz; subjects 1–24 / 25–28 / 29–32 |

REFED requires the numeric subject folders, `annotations/` and `SAM_score.csv`.
`--annotations` and `--sam` override the latter two paths. The supplied channel
metadata pickle files are included beside `build_samples.py`. Intermediate MATLAB
files are written under `OUTPUT/intermediate/`. `refed/annotations.py` contains the
original annotation readers. `refed/plot_ratings.py` is the optional rating-plot
utility and is not part of the training-data pipeline.

## Preserved numerical details

- DEAP retains FIR bandpass filtering (EEG 0.05–60 Hz, EOG 0.05–50 Hz,
  EMG 10–60 Hz) and the original 50 Hz notch before resampling.
- HCI uses the sampling rate from the recording header, filters EEG at 0.1–75 Hz,
  ECG at 0.05–95 Hz and GSR at 0.05–5 Hz, then resamples the cropped experimental
  signal to 200 Hz. The baseline is computed as in the supplied code; it is not
  subtracted from the saved experimental samples.
- MER retains its modality filters (EEG 0.1–80 Hz, GSR 0.05–1.95 Hz,
  PPG 0.05–20 Hz), notch attempt and first segment per stimulus ID. The original
  handling of an unavailable notch is preserved.
- REFED EEG filtering and notch calls were commented out in the supplied script;
  the active EEG path resamples directly. The fNIRS path applies a 0.01–10 Hz IIR
  filter, resamples first to 25 Hz and then to 200 Hz, and discards incomplete
  trailing windows. These operations have not been replaced with a new protocol.
- For every dataset, each channel's normalization mean is the **average of the
  training windows' temporal means**; its scale is the **average of the training
  windows' temporal standard deviations**. All splits use these training-derived
  values with epsilon `1e-8`. This is not a pooled standard deviation over all
  training points. Only signal samples are normalized by the normalization stage.

Library versions affect filter design and resampling. Keep the original runtime
versions when reproducing historical results. The packaged HCI implementation
matches the corrected source with actual 200 Hz resampling.

## Labels and training integration

The original label arrays remain unchanged:

| Dataset | Stored labels | Binary threshold |
|---|---|---|
| DEAP | valence, arousal | score ≥ 5 |
| HCI | valence, arousal | original `int(score / 5)` |
| MER | valence, arousal | score > 5 |
| REFED | arousal, valence, dynamic valence, dynamic arousal | first two: original SAM `score // 5`; dynamic annotations retain the original `/256` conversion |

The entry point adds `dataset_info.label_names` to the manifests. FIPT uses these
names to resolve the requested prediction target. Legacy manifests without this
field retain their configured label index. This avoids changing stored labels while
making the dataset-specific ordering explicit.

Completed outputs contain `data/`, `train.json`, `val.json`, `test.json` and the
run record. Use the output directory directly for training:

```bash
python run_finetuning.py --dataset HCI --backbone CBraMod --target arousal --data-root /data/FIPT_ready/HCI
```

The output manifests contain absolute sample paths, matching the original format.
Prepare data on the machine that will train, or update manifest paths after moving
the output directory.

## Validation

`source_hashes.json` identifies the supplied source files. Synthetic comparisons
against the original scripts verified all four normalization stages, HCI splitting,
DEAP filtering/resampling/labels, HCI filtering and actual resampling, MER segmentation
and window alignment, and REFED filtering/resampling/windowing. Checked signal arrays
were exactly equal in the same local runtime. See `validation.json` for coverage.
Full reprocessing of all four raw datasets has not been run. The packaging does not
establish agreement between these scripts and every historical experiment.
