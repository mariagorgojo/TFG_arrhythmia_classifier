from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "occurrence_dataset" / "occurrences_manifest.csv"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "training_dataset" / "ecg_training_dataset.npz"
)
DEFAULT_TARGET_SAMPLES = 2048
DEFAULT_REQUIRED_SAMPLING_RATE_HZ = 128.0
RHYTHM_FEATURE_NAMES = (
    "waveform_duration_seconds",
    "marker_count",
    "rr_count",
    "rr_mean_seconds",
    "rr_median_seconds",
    "rr_std_seconds",
    "rr_cv",
    "rr_min_seconds",
    "rr_max_seconds",
    "heart_rate_mean_bpm",
    "heart_rate_median_bpm",
    "marker_vs_fraction",
    "marker_bs_fraction",
    "marker_vts_fraction",
    "marker_vfs_fraction",
    "marker_reject_fraction",
    "raw_signal_std_mv",
    "raw_signal_peak_to_peak_mv",
)
DEFAULT_LABELS = (
    "CurrentECG",
    "AF",
    "Asystole",
    "Brady",
    "VT",
    "FVT",
    "PatientActivated",
)
LABEL_STRATEGIES = {
    "raw7": {
        "labels": DEFAULT_LABELS,
        "mapping": {label: label for label in DEFAULT_LABELS},
    },
    "clinical5": {
        "labels": ("CurrentECG", "AF", "Asystole", "Brady", "VT"),
        "mapping": {
            "CurrentECG": "CurrentECG",
            "AF": "AF",
            "Asystole": "Asystole",
            "Brady": "Brady",
            "VT": "VT",
            "FVT": "VT",
        },
    },
    "arrhythmia4": {
        "labels": ("AF", "Asystole", "Brady", "VT"),
        "mapping": {
            "AF": "AF",
            "Asystole": "Asystole",
            "Brady": "Brady",
            "VT": "VT",
            "FVT": "VT",
        },
    },
}


@dataclass(slots=True)
class ManifestOccurrence:
    occurrence_id: str
    occurrence_type: str
    device_id: str
    sampling_rate_hz: float | None
    sample_count: int
    has_stored_waveform: bool
    waveform_npz_path: Path


@dataclass(slots=True)
class PreparedDatasetSummary:
    output_path: Path
    selected_count: int
    target_samples: int
    label_counts: dict[str, int]
    label_to_index: dict[str, int]
    skipped_count: int


def _bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _float_from_csv(value: str) -> float | None:
    if value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_from_csv(value: str) -> int:
    if value.strip() == "":
        return 0
    return int(float(value))


def _resolve_path(path_text: str, manifest_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path

    project_relative = manifest_path.resolve().parents[3] / path
    if project_relative.exists():
        return project_relative

    return path


def read_manifest(manifest_path: Path) -> list[ManifestOccurrence]:
    rows: list[ManifestOccurrence] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ManifestOccurrence(
                    occurrence_id=row["occurrence_id"],
                    occurrence_type=row["occurrence_type"],
                    device_id=row["device_id"],
                    sampling_rate_hz=_float_from_csv(row["sampling_rate_hz"]),
                    sample_count=_int_from_csv(row["sample_count"]),
                    has_stored_waveform=_bool_from_csv(row["has_stored_waveform"]),
                    waveform_npz_path=_resolve_path(
                        row["waveform_npz_path"], manifest_path
                    ),
                )
            )
    return rows


def filter_occurrences(
    rows: list[ManifestOccurrence],
    *,
    label_mapping: dict[str, str],
    required_sampling_rate_hz: float | None,
    min_samples: int,
) -> list[ManifestOccurrence]:
    filtered: list[ManifestOccurrence] = []
    for row in rows:
        if row.occurrence_type not in label_mapping:
            continue
        if not row.has_stored_waveform:
            continue
        if row.sample_count < min_samples:
            continue
        if required_sampling_rate_hz is not None and row.sampling_rate_hz != required_sampling_rate_hz:
            continue
        filtered.append(row)
    return filtered


def select_balanced_rows(
    rows: list[ManifestOccurrence],
    *,
    max_per_label: int | None,
) -> list[ManifestOccurrence]:
    if max_per_label is None:
        return rows

    selected: list[ManifestOccurrence] = []
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if counts[row.occurrence_type] >= max_per_label:
            continue
        selected.append(row)
        counts[row.occurrence_type] += 1
    return selected


def crop_or_pad_signal(signal: np.ndarray, target_samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size >= target_samples:
        start = (signal.size - target_samples) // 2
        return signal[start : start + target_samples]

    output = np.zeros(target_samples, dtype=np.float32)
    start = (target_samples - signal.size) // 2
    output[start : start + signal.size] = signal
    return output


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std == 0.0:
        return signal - mean
    return (signal - mean) / std


def load_prepared_signal(row: ManifestOccurrence, target_samples: int) -> np.ndarray:
    arrays = np.load(row.waveform_npz_path)
    signal = arrays["waveform_values"].astype(np.float32)
    signal = crop_or_pad_signal(signal, target_samples)
    return normalize_signal(signal)


def extract_rhythm_features(arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    signal = arrays["waveform_values"].astype(np.float32)
    time_seconds = arrays["time_seconds"].astype(np.float32)
    rr_intervals = arrays["rr_intervals_seconds"].astype(np.float32)
    marker_names = arrays["marker_names"].astype(str)

    valid_rr = rr_intervals[np.isfinite(rr_intervals) & (rr_intervals > 0)]
    if valid_rr.size:
        rr_mean = float(np.mean(valid_rr))
        rr_median = float(np.median(valid_rr))
        rr_std = float(np.std(valid_rr))
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0.0
        rr_min = float(np.min(valid_rr))
        rr_max = float(np.max(valid_rr))
        heart_rate_mean = 60.0 / rr_mean if rr_mean > 0 else 0.0
        heart_rate_median = 60.0 / rr_median if rr_median > 0 else 0.0
    else:
        rr_mean = rr_median = rr_std = rr_cv = rr_min = rr_max = 0.0
        heart_rate_mean = heart_rate_median = 0.0

    marker_count = int(marker_names.size)
    if marker_count:
        upper_markers = np.char.upper(marker_names)
        vs_fraction = float(np.mean(upper_markers == "VS"))
        bs_fraction = float(np.mean(upper_markers == "BS"))
        vts_fraction = float(np.mean(upper_markers == "VTS"))
        vfs_fraction = float(np.mean(upper_markers == "VFS"))
        reject_fraction = float(
            np.mean(np.char.find(upper_markers.astype(str), "REJECT") >= 0)
        )
    else:
        vs_fraction = bs_fraction = vts_fraction = vfs_fraction = reject_fraction = 0.0

    duration = float(time_seconds[-1] - time_seconds[0]) if time_seconds.size >= 2 else 0.0
    signal_std = float(np.std(signal)) if signal.size else 0.0
    signal_peak_to_peak = float(np.ptp(signal)) if signal.size else 0.0

    return np.asarray(
        [
            duration,
            float(marker_count),
            float(valid_rr.size),
            rr_mean,
            rr_median,
            rr_std,
            rr_cv,
            rr_min,
            rr_max,
            heart_rate_mean,
            heart_rate_median,
            vs_fraction,
            bs_fraction,
            vts_fraction,
            vfs_fraction,
            reject_fraction,
            signal_std,
            signal_peak_to_peak,
        ],
        dtype=np.float32,
    )


def load_prepared_signal_and_features(
    row: ManifestOccurrence,
    target_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(row.waveform_npz_path)
    signal = arrays["waveform_values"].astype(np.float32)
    rhythm_features = extract_rhythm_features(arrays)
    signal = crop_or_pad_signal(signal, target_samples)
    return normalize_signal(signal), rhythm_features


def build_training_dataset(
    manifest_path: Path,
    output_path: Path,
    *,
    target_samples: int = DEFAULT_TARGET_SAMPLES,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    label_mapping: dict[str, str] | None = None,
    label_strategy: str = "raw7",
    required_sampling_rate_hz: float | None = DEFAULT_REQUIRED_SAMPLING_RATE_HZ,
    min_samples: int = 256,
    max_per_label: int | None = 5000,
) -> PreparedDatasetSummary:
    rows = read_manifest(manifest_path)
    if label_mapping is None:
        label_mapping = {label: label for label in labels}
    filtered = filter_occurrences(
        rows,
        label_mapping=label_mapping,
        required_sampling_rate_hz=required_sampling_rate_hz,
        min_samples=min_samples,
    )
    selected = select_balanced_rows(filtered, max_per_label=max_per_label)

    label_to_index = {label: index for index, label in enumerate(labels)}
    X = np.zeros((len(selected), target_samples), dtype=np.float32)
    rhythm_features = np.zeros((len(selected), len(RHYTHM_FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(len(selected), dtype=np.int16)
    occurrence_ids: list[str] = []
    device_ids: list[str] = []
    label_names: list[str] = []
    source_label_names: list[str] = []

    skipped = 0
    kept_index = 0
    for row in selected:
        mapped_label = label_mapping[row.occurrence_type]
        try:
            X[kept_index], rhythm_features[kept_index] = (
                load_prepared_signal_and_features(row, target_samples)
            )
        except (FileNotFoundError, KeyError, OSError, ValueError):
            skipped += 1
            continue
        y[kept_index] = label_to_index[mapped_label]
        occurrence_ids.append(row.occurrence_id)
        device_ids.append(row.device_id)
        label_names.append(mapped_label)
        source_label_names.append(row.occurrence_type)
        kept_index += 1

    X = X[:kept_index]
    rhythm_features = rhythm_features[:kept_index]
    y = y[:kept_index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X,
        rhythm_features=rhythm_features,
        rhythm_feature_names=np.asarray(RHYTHM_FEATURE_NAMES, dtype="U64"),
        y=y,
        occurrence_ids=np.asarray(occurrence_ids, dtype="U128"),
        device_ids=np.asarray(device_ids, dtype="U64"),
        label_names=np.asarray(label_names, dtype="U32"),
        source_label_names=np.asarray(source_label_names, dtype="U32"),
        labels=np.asarray(labels, dtype="U32"),
        target_samples=np.asarray([target_samples], dtype=np.int32),
        required_sampling_rate_hz=np.asarray(
            [np.nan if required_sampling_rate_hz is None else required_sampling_rate_hz],
            dtype=np.float32,
        ),
    )

    summary = PreparedDatasetSummary(
        output_path=output_path,
        selected_count=kept_index,
        target_samples=target_samples,
        label_counts=dict(Counter(label_names)),
        label_to_index=label_to_index,
        skipped_count=skipped,
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "output_path": str(summary.output_path),
                "selected_count": summary.selected_count,
                "target_samples": summary.target_samples,
                "label_counts": summary.label_counts,
                "label_to_index": summary.label_to_index,
                "rhythm_feature_names": list(RHYTHM_FEATURE_NAMES),
                "skipped_count": summary.skipped_count,
                "filters": {
                    "labels": labels,
                    "label_strategy": label_strategy,
                    "label_mapping": label_mapping,
                    "required_sampling_rate_hz": required_sampling_rate_hz,
                    "min_samples": min_samples,
                    "max_per_label": max_per_label,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-length ECG arrays for neural-network training."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Occurrence manifest CSV. Default: {DEFAULT_MANIFEST_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Training dataset output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=DEFAULT_TARGET_SAMPLES,
        help=f"Fixed ECG length after crop/pad. Default: {DEFAULT_TARGET_SAMPLES}",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Occurrence labels to include.",
    )
    parser.add_argument(
        "--label-strategy",
        choices=sorted(LABEL_STRATEGIES),
        default="raw7",
        help=(
            "Label mapping strategy. raw7 keeps original selected labels; "
            "clinical5 keeps CurrentECG as a first-iteration class; "
            "arrhythmia4 excludes CurrentECG and keeps only diagnostic arrhythmia labels."
        ),
    )
    parser.add_argument(
        "--required-sampling-rate-hz",
        type=float,
        default=DEFAULT_REQUIRED_SAMPLING_RATE_HZ,
        help="Only include this sampling rate. Use 0 to disable this filter.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=256,
        help="Minimum stored waveform sample count.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=5000,
        help="Maximum examples per label. Use 0 to include all matching examples.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    required_sampling_rate_hz = (
        None if args.required_sampling_rate_hz == 0 else args.required_sampling_rate_hz
    )
    max_per_label = None if args.max_per_label == 0 else args.max_per_label
    strategy = LABEL_STRATEGIES[args.label_strategy]
    labels = tuple(args.labels)
    label_mapping = {label: label for label in labels}
    if args.label_strategy != "raw7":
        labels = strategy["labels"]
        label_mapping = strategy["mapping"]
    summary = build_training_dataset(
        args.manifest_path,
        args.output_path,
        target_samples=args.target_samples,
        labels=labels,
        label_mapping=label_mapping,
        label_strategy=args.label_strategy,
        required_sampling_rate_hz=required_sampling_rate_hz,
        min_samples=args.min_samples,
        max_per_label=max_per_label,
    )

    print(f"Output dataset: {summary.output_path}")
    print(f"Selected examples: {summary.selected_count}")
    print(f"Target samples: {summary.target_samples}")
    print(f"Skipped while loading: {summary.skipped_count}")
    print("Labels:")
    for label, count in sorted(summary.label_counts.items()):
        print(f" - {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
