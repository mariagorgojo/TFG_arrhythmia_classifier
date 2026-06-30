from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".pythonlibs"

if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(LOCAL_SITE_PACKAGES))

import numpy as np


DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "training_dataset"
    / "ecg_training_dataset_currentecg5_features.npz"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "ecg_training_dataset_currentecg5_features_pseudonymized.npz"
)
DATASET_SCOPE = "currentecg5_pseudonymized_candidate"
REQUIRED_KEYS = (
    "X",
    "rhythm_features",
    "rhythm_feature_names",
    "y",
    "device_ids",
    "labels",
    "target_samples",
    "required_sampling_rate_hz",
)
REMOVED_FIELDS = (
    "occurrence_ids",
    "label_names",
    "source_label_names",
)


@dataclass(frozen=True)
class ExportSummary:
    output_file: str
    dataset_scope: str
    examples: int
    signal_shape: list[int]
    rhythm_features_shape: list[int]
    device_group_count: int
    labels: list[str]
    label_counts: dict[str, int]
    removed_fields: list[str]
    retained_fields: list[str]
    privacy_note: str


def replace_device_ids_with_groups(device_ids: np.ndarray) -> np.ndarray:
    unique_ids = sorted(set(device_ids.astype(str).tolist()))
    mapping = {
        device_id: f"group_{index:04d}"
        for index, device_id in enumerate(unique_ids, start=1)
    }
    return np.asarray([mapping[str(device_id)] for device_id in device_ids], dtype="U16")


def _as_scalar_array(source: np.lib.npyio.NpzFile, key: str, dtype: type) -> np.ndarray:
    value = source[key]
    if value.shape == ():
        value = np.asarray([value.item()])
    return value.astype(dtype)


def export_dataset(input_path: Path, output_path: Path) -> ExportSummary:
    source = np.load(input_path, allow_pickle=False)
    missing_keys = [key for key in REQUIRED_KEYS if key not in source]
    if missing_keys:
        raise KeyError(f"Missing required dataset keys: {missing_keys}")

    labels = source["labels"].astype(str)
    y = source["y"].astype(np.int16)
    grouped_device_ids = replace_device_ids_with_groups(source["device_ids"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=source["X"].astype(np.float32),
        rhythm_features=source["rhythm_features"].astype(np.float32),
        rhythm_feature_names=source["rhythm_feature_names"].astype("U64"),
        y=y,
        device_ids=grouped_device_ids,
        labels=labels.astype("U32"),
        target_samples=_as_scalar_array(source, "target_samples", np.int32),
        required_sampling_rate_hz=_as_scalar_array(
            source, "required_sampling_rate_hz", np.float32
        ),
        dataset_scope=np.asarray([DATASET_SCOPE], dtype="U64"),
    )

    label_counts = Counter(labels[label_index] for label_index in y)
    retained_fields = [
        "X",
        "rhythm_features",
        "rhythm_feature_names",
        "y",
        "device_ids",
        "labels",
        "target_samples",
        "required_sampling_rate_hz",
        "dataset_scope",
    ]
    summary = ExportSummary(
        output_file=output_path.name,
        dataset_scope=DATASET_SCOPE,
        examples=int(y.size),
        signal_shape=list(source["X"].shape),
        rhythm_features_shape=list(source["rhythm_features"].shape),
        device_group_count=int(np.unique(grouped_device_ids).size),
        labels=labels.tolist(),
        label_counts=dict(label_counts),
        removed_fields=[field for field in REMOVED_FIELDS if field in source],
        retained_fields=retained_fields,
        privacy_note=(
            "Original occurrence identifiers were removed and device identifiers "
            "were replaced with generic groups. ECG waveforms and rhythm features "
            "remain clinical data. Obtain data-governance approval before uploading "
            "or sharing this file publicly."
        ),
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary.__dict__, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a pseudonymized five-class CurrentECG training dataset candidate."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_dataset(args.input_path, args.output_path)
    print(f"Output dataset: {summary.output_file}")
    print(f"Examples: {summary.examples}")
    print(f"Device groups: {summary.device_group_count}")
    print("Review data-governance requirements before uploading this file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
