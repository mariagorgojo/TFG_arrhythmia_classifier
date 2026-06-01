from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".pythonlibs"

if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(LOCAL_SITE_PACKAGES))
for dll_dir in (
    LOCAL_SITE_PACKAGES / "torch" / "lib",
    LOCAL_SITE_PACKAGES / "numpy.libs",
    LOCAL_SITE_PACKAGES / "scipy.libs",
):
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from arrhythmia_classifier.dataset_loader import (
    ManifestOccurrence,
    load_prepared_signal_and_features,
    read_manifest,
)
from arrhythmia_classifier.hybrid_cnn1d_classifier import ECGHybridCNN

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "occurrence_dataset" / "occurrences_manifest.csv"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "hybrid_cnn1d_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "currentecg_exploratory"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def select_currentecg_rows(
    manifest_path: Path,
    *,
    required_sampling_rate_hz: float,
    min_samples: int,
) -> tuple[list[ManifestOccurrence], dict[str, int]]:
    all_rows = read_manifest(manifest_path)
    selected: list[ManifestOccurrence] = []
    excluded = Counter()

    for row in all_rows:
        if row.occurrence_type != "CurrentECG":
            continue
        if not row.has_stored_waveform:
            excluded["without_stored_waveform"] += 1
            continue
        if row.sample_count < min_samples:
            excluded["too_short"] += 1
            continue
        if row.sampling_rate_hz != required_sampling_rate_hz:
            excluded["different_sampling_rate"] += 1
            continue
        selected.append(row)

    return selected, dict(excluded)


def standardize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    safe_std = np.where(std == 0, 1.0, std)
    return ((features - mean) / safe_std).astype(np.float32)


def predict_batch(
    model: ECGHybridCNN,
    signals: list[np.ndarray],
    rhythm_features: list[np.ndarray],
    *,
    rhythm_mean: np.ndarray,
    rhythm_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    signal_tensor = torch.tensor(
        np.asarray(signals, dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(1).to(device)
    feature_values = standardize_features(
        np.asarray(rhythm_features, dtype=np.float32),
        rhythm_mean,
        rhythm_std,
    )
    feature_tensor = torch.tensor(feature_values, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(signal_tensor, feature_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predicted_indices = np.argmax(probabilities, axis=1)
    return predicted_indices, probabilities


def write_distribution_csv(
    path: Path,
    labels: list[str],
    predicted_counts: Counter[str],
    total: int,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted_label", "count", "percentage"])
        for label in labels:
            count = predicted_counts[label]
            percentage = (count / total * 100.0) if total else 0.0
            writer.writerow([label, count, f"{percentage:.4f}"])


def write_confidence_csv(
    path: Path,
    confidence_counts: Counter[str],
    total: int,
) -> None:
    bands = ["<0.50", "0.50-0.69", "0.70-0.89", ">=0.90"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["confidence_band", "count", "percentage"])
        for band in bands:
            count = confidence_counts[band]
            percentage = (count / total * 100.0) if total else 0.0
            writer.writerow([band, count, f"{percentage:.4f}"])


def confidence_band(confidence: float) -> str:
    if confidence < 0.50:
        return "<0.50"
    if confidence < 0.70:
        return "0.50-0.69"
    if confidence < 0.90:
        return "0.70-0.89"
    return ">=0.90"


def plot_prediction_distribution(
    path: Path,
    labels: list[str],
    predicted_counts: Counter[str],
    total: int,
) -> None:
    width = 1050
    height = 650
    left = 155
    top = 145
    right = 970
    bottom = 545
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(17)
    label_font = load_font(18)
    value_font = load_font(17, bold=True)

    draw.text((35, 25), "CurrentECG exploratory prediction distribution", fill="#172033", font=title_font)
    draw.text(
        (35, 68),
        "Forced-choice predictions: NSR/normal and other rhythms are not available model outputs.",
        fill="#47515f",
        font=subtitle_font,
    )

    max_count = max([predicted_counts[label] for label in labels] + [1])
    bar_height = 58
    gap = 31
    colors = ["#2d6cdf", "#5b8def", "#8aaef5", "#1c4a9e"]

    for index, label in enumerate(labels):
        y = top + index * (bar_height + gap)
        count = predicted_counts[label]
        percentage = (count / total * 100.0) if total else 0.0
        bar_width = int((right - left) * (count / max_count))
        draw.text((40, y + 17), label, fill="#172033", font=label_font)
        draw.rectangle((left, y, right, y + bar_height), fill="#eef3fa")
        draw.rectangle((left, y, left + bar_width, y + bar_height), fill=colors[index % len(colors)])
        draw.text(
            (min(left + bar_width + 12, right - 135), y + 17),
            f"{count:,} ({percentage:.2f}%)",
            fill="#172033",
            font=value_font,
        )

    draw.text(
        (35, bottom + 45),
        f"Total usable CurrentECG records: {total:,}",
        fill="#47515f",
        font=subtitle_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def run_inference(
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    required_sampling_rate_hz: float = 128.0,
    min_samples: int = 256,
    batch_size: int = 256,
    progress_every: int = 5000,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    labels = [str(label) for label in checkpoint["labels"]]
    target_samples = int(checkpoint["target_samples"])
    rhythm_feature_names = [str(name) for name in checkpoint["rhythm_feature_names"]]
    rhythm_mean = np.asarray(checkpoint["rhythm_feature_mean"], dtype=np.float32)
    rhythm_std = np.asarray(checkpoint["rhythm_feature_std"], dtype=np.float32)

    model = ECGHybridCNN(
        num_classes=len(labels),
        num_rhythm_features=len(rhythm_feature_names),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    selected_rows, excluded_counts = select_currentecg_rows(
        manifest_path,
        required_sampling_rate_hz=required_sampling_rate_hz,
        min_samples=min_samples,
    )

    predictions_path = output_dir / "currentecg_predictions.csv"
    predicted_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    confidence_values: list[float] = []
    skipped_loading = 0
    processed = 0

    fieldnames = [
        "occurrence_id",
        "device_id",
        "predicted_label",
        "confidence",
    ] + [f"probability_{label}" for label in labels]

    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        pending_rows: list[ManifestOccurrence] = []
        pending_signals: list[np.ndarray] = []
        pending_features: list[np.ndarray] = []

        def flush_batch() -> None:
            nonlocal processed
            if not pending_rows:
                return
            predicted_indices, probabilities = predict_batch(
                model,
                pending_signals,
                pending_features,
                rhythm_mean=rhythm_mean,
                rhythm_std=rhythm_std,
                device=device,
            )
            for row, label_index, row_probabilities in zip(
                pending_rows,
                predicted_indices,
                probabilities,
            ):
                predicted_label = labels[int(label_index)]
                confidence = float(np.max(row_probabilities))
                predicted_counts[predicted_label] += 1
                confidence_counts[confidence_band(confidence)] += 1
                confidence_values.append(confidence)
                output_row: dict[str, object] = {
                    "occurrence_id": row.occurrence_id,
                    "device_id": row.device_id,
                    "predicted_label": predicted_label,
                    "confidence": f"{confidence:.6f}",
                }
                for label, probability in zip(labels, row_probabilities):
                    output_row[f"probability_{label}"] = f"{float(probability):.6f}"
                writer.writerow(output_row)
                processed += 1

            pending_rows.clear()
            pending_signals.clear()
            pending_features.clear()

        for row_index, row in enumerate(selected_rows, start=1):
            try:
                signal, features = load_prepared_signal_and_features(row, target_samples)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                skipped_loading += 1
                continue
            pending_rows.append(row)
            pending_signals.append(signal)
            pending_features.append(features)
            if len(pending_rows) >= batch_size:
                flush_batch()
            if progress_every and row_index % progress_every == 0:
                print(
                    f"Processed {processed:,} / {len(selected_rows):,} selected CurrentECG records",
                    flush=True,
                )
        flush_batch()

    distribution_path = output_dir / "prediction_distribution.csv"
    write_distribution_csv(distribution_path, labels, predicted_counts, processed)
    confidence_path = output_dir / "confidence_distribution.csv"
    write_confidence_csv(confidence_path, confidence_counts, processed)
    chart_path = output_dir / "prediction_distribution.png"
    plot_prediction_distribution(chart_path, labels, predicted_counts, processed)

    confidence_array = np.asarray(confidence_values, dtype=np.float32)
    summary = {
        "analysis_type": "unlabeled_currentecg_exploratory_inference",
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "labels": labels,
        "target_samples": target_samples,
        "required_sampling_rate_hz": required_sampling_rate_hz,
        "min_samples": min_samples,
        "selected_currentecg_records": len(selected_rows),
        "processed_currentecg_records": processed,
        "skipped_while_loading": skipped_loading,
        "excluded_before_loading": excluded_counts,
        "predicted_counts": dict(predicted_counts),
        "predicted_percentages": {
            label: (predicted_counts[label] / processed * 100.0) if processed else 0.0
            for label in labels
        },
        "confidence_counts": dict(confidence_counts),
        "mean_confidence": float(np.mean(confidence_array)) if confidence_array.size else 0.0,
        "median_confidence": float(np.median(confidence_array)) if confidence_array.size else 0.0,
        "limitations": [
            "CurrentECG is a device/export label, not a confirmed clinical diagnosis.",
            "Accuracy, sensitivity, F1-score and a confusion matrix cannot be calculated for this cohort without reference labels.",
            "The model has no NSR/normal or other/indeterminate class and is forced to choose one of four arrhythmias for every record.",
            "AF predictions may include NSR or unsupported rhythms that appear closest to AF because of slight irregularity, noise, artifact or oversensing.",
            "These predictions are exploratory and require expert clinical review before clinical interpretation.",
        ],
        "outputs": {
            "predictions_csv": str(predictions_path),
            "prediction_distribution_csv": str(distribution_path),
            "confidence_distribution_csv": str(confidence_path),
            "prediction_distribution_png": str(chart_path),
        },
        "device": str(device),
    }
    summary_path = output_dir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "CurrentECG exploratory inference report",
        "",
        "Important: CurrentECG is not a confirmed diagnostic label.",
        "These predictions are exploratory and cannot be evaluated with accuracy or F1-score without clinical reference labels.",
        "The model has no NSR/normal or other/indeterminate output, so every CurrentECG is forced into one of four arrhythmia classes.",
        "A predicted AF result may therefore be a wrong nearest-class assignment, particularly for NSR-like records with slight irregularity, noise, artifact or oversensing.",
        "",
        f"Processed CurrentECG records: {processed:,}",
        f"Skipped while loading: {skipped_loading:,}",
        "",
        "Predicted distribution:",
    ]
    for label in labels:
        count = predicted_counts[label]
        percentage = count / processed * 100.0 if processed else 0.0
        report_lines.append(f"- {label}: {count:,} ({percentage:.2f}%)")
    report_lines.extend(
        [
            "",
            f"Mean confidence: {summary['mean_confidence']:.4f}",
            f"Median confidence: {summary['median_confidence']:.4f}",
        ]
    )
    (output_dir / "inference_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exploratory hybrid-CNN inference on unlabeled CurrentECG records."
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--required-sampling-rate-hz", type=float, default=128.0)
    parser.add_argument("--min-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_inference(
        args.manifest_path,
        args.model_path,
        args.output_dir,
        required_sampling_rate_hz=args.required_sampling_rate_hz,
        min_samples=args.min_samples,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )
    print(f"Processed CurrentECG records: {summary['processed_currentecg_records']:,}")
    print(f"Summary: {args.output_dir / 'inference_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
