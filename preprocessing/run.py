"""Run one dataset's original preprocessing stages in sequence.

Usage: python -m preprocessing.run --dataset HCI --source /data/Sessions --output /data/HCI_ready
"""

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

STAGES = {
    "DEAP": ("prepare", "normalize"),
    "HCI": ("prepare", "normalize"),
    "MER": ("prepare", "normalize"),
    "REFED": ("prepare_signals", "build_samples", "normalize"),
}
LABEL_NAMES = {
    "DEAP": ["valence", "arousal"],
    "HCI": ["valence", "arousal"],
    "MER": ["valence", "arousal"],
    "REFED": ["arousal", "valence", "dynamic_valence", "dynamic_arousal"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=STAGES, required=True)
    parser.add_argument(
        "--source", type=Path, required=True, help="Original dataset input directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for samples and manifests",
    )
    parser.add_argument("--labels", type=Path, help="MER Raw_data directory")
    parser.add_argument("--ratings", type=Path, help="DEAP participant_ratings.csv")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="REFED annotation directory; default: SOURCE/annotations",
    )
    parser.add_argument(
        "--sam", type=Path, help="REFED SAM_score.csv; default: SOURCE/SAM_score.csv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or source in output.parents:
        raise ValueError("Choose an output directory outside the raw dataset tree.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Output is not empty. Use a new directory to avoid overwriting or normalizing twice."
        )
    settings = SimpleNamespace(
        source=source.as_posix(),
        output=output.as_posix(),
        intermediate=(output / "intermediate").as_posix(),
        labels=args.labels.resolve().as_posix() if args.labels else None,
        ratings=(args.ratings or source.parent / "participant_ratings.csv")
        .resolve()
        .as_posix(),
        annotations=(args.annotations or source / "annotations").resolve().as_posix(),
        sam=(args.sam or source / "SAM_score.csv").resolve().as_posix(),
    )
    if args.dataset == "MER" and (
        not settings.labels or not Path(settings.labels).is_dir()
    ):
        raise ValueError("MER requires --labels pointing to its Raw_data directory.")
    if args.dataset == "DEAP" and not Path(settings.ratings).is_file():
        raise FileNotFoundError(
            "Supply --ratings with participant_ratings.csv (read by the original script)."
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir()
    record = dict(
        dataset=args.dataset,
        paths=vars(settings),
        completed_stages=[],
        status="running",
    )
    record_path = output / "preprocessing_run.json"
    try:
        for stage in STAGES[args.dataset]:
            record["current_stage"] = stage
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"{args.dataset}: {stage}", flush=True)
            module = importlib.import_module(
                f"preprocessing.{args.dataset.lower()}.{stage}"
            )
            module.run(settings)
            record["completed_stages"].append(stage)
        for split in ("train", "val", "test"):
            manifest_path = output / f"{split}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_info"]["label_names"] = LABEL_NAMES[args.dataset]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Completed: {output}. Use this directory as the training --data-root.")


if __name__ == "__main__":
    main()
