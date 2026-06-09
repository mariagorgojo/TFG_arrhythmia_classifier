from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from arrhythmia_classifier.hybrid_cnn1d_currentecg5_optuna import TunableHybridCNN
from arrhythmia_classifier.hybrid_cnn1d_train_val_test import (
    split_train_validation_test_by_device,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_dataset"
    / "ecg_training_dataset_currentecg5_features.npz"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "hybrid_cnn1d_currentecg5_optuna"
    / "hybrid_cnn1d_currentecg5_optuna_model.pt"
)
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "results" / "hybrid_cnn1d_currentecg5_optuna" / "metrics.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "currentecg_confidence_analysis"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def stable_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_model(checkpoint: dict[str, object], num_rhythm_features: int, device: torch.device) -> TunableHybridCNN:
    labels = checkpoint["labels"]
    params = checkpoint["hyperparameters"]
    model = TunableHybridCNN(
        num_classes=len(labels),
        num_rhythm_features=num_rhythm_features,
        conv_layers=int(params["conv_layers"]),
        base_channels=int(params["base_channels"]),
        kernel_size=int(params["kernel_size"]),
        rhythm_hidden=int(params["rhythm_hidden"]),
        dropout=float(params["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def select_subset(
    *,
    subset: str,
    y: np.ndarray,
    labels: np.ndarray,
    device_ids: np.ndarray,
    metrics: dict[str, object],
) -> np.ndarray:
    current_index = int(np.where(labels == "CurrentECG")[0][0])
    if subset == "all_currentecg":
        return np.where(y == current_index)[0]
    train_index, validation_index, test_index = split_train_validation_test_by_device(
        np.zeros((y.shape[0], 1), dtype=np.float32),
        y,
        device_ids,
        validation_size=float(metrics.get("validation_size_requested", 0.2)),
        test_size=float(metrics.get("test_size_requested", 0.2)),
        random_state=int(metrics.get("random_state", 42)),
        split_strategy=str(metrics.get("split_strategy", "stratified_group")),
    )
    if subset == "test_currentecg":
        return test_index[y[test_index] == current_index]
    if subset == "validation_currentecg":
        return validation_index[y[validation_index] == current_index]
    if subset == "train_currentecg":
        return train_index[y[train_index] == current_index]
    raise ValueError(f"Unsupported subset: {subset}")


def predict_probabilities(
    *,
    model: torch.nn.Module,
    X: np.ndarray,
    rhythm_features: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            ecg_batch = torch.tensor(X[batch_indices], dtype=torch.float32).unsqueeze(1).to(device)
            rhythm_batch = torch.tensor(
                rhythm_features[batch_indices], dtype=torch.float32
            ).to(device)
            logits = model(ecg_batch, rhythm_batch)
            probabilities.append(softmax_logits(logits).cpu().numpy())
    return np.vstack(probabilities)


def probability_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / math.log(probabilities.shape[1])


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_case_rows(
    *,
    indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    occurrence_ids: np.ndarray,
    device_ids: np.ndarray,
    current_label_index: int,
    order: np.ndarray,
    top_n: int,
) -> list[dict[str, object]]:
    non_current_indices = [i for i in range(len(labels)) if i != current_label_index]
    rows: list[dict[str, object]] = []
    for rank, probability_row_index in enumerate(order[:top_n], start=1):
        original_index = int(indices[probability_row_index])
        row_probabilities = probabilities[probability_row_index]
        predicted_index = int(np.argmax(row_probabilities))
        alternative_index = max(
            non_current_indices,
            key=lambda label_index: row_probabilities[label_index],
        )
        row = {
            "rank": rank,
            "dataset_index": original_index,
            "occurrence_hash": stable_hash(occurrence_ids[original_index]),
            "device_hash": stable_hash(device_ids[original_index]),
            "predicted_label": str(labels[predicted_index]),
            "top_non_current_label": str(labels[alternative_index]),
            "p_currentecg": float(row_probabilities[current_label_index]),
            "p_top_non_current": float(row_probabilities[alternative_index]),
            "confidence_margin": float(
                row_probabilities[current_label_index] - row_probabilities[alternative_index]
            ),
            "entropy": float(entropy(row_probabilities.reshape(1, -1))[0]),
        }
        for label_index, label in enumerate(labels):
            row[f"p_{label}"] = float(row_probabilities[label_index])
        rows.append(row)
    return rows


def write_histogram_csv(path: Path, values: np.ndarray, *, bin_width: float = 0.05) -> list[dict[str, object]]:
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    counts, edges = np.histogram(values, bins=bins)
    rows = []
    total = int(values.size)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        rows.append(
            {
                "bin_left": round(float(left), 4),
                "bin_right": round(float(right), 4),
                "count": int(count),
                "percentage": float(count / total) if total else 0.0,
            }
        )
    write_csv(path, rows, ["bin_left", "bin_right", "count", "percentage"])
    return rows


def plot_histogram(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    width, height = 980, 560
    margin_left, margin_right = 80, 30
    margin_top, margin_bottom = 70, 95
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(26, bold=True)
    axis_font = load_font(15, bold=True)
    tick_font = load_font(12)
    draw.text((30, 25), title, fill="#172033", font=title_font)
    max_count = max(int(row["count"]) for row in rows) if rows else 1
    max_count = max(max_count, 1)
    bar_width = plot_width / max(len(rows), 1)
    axis_x = margin_left
    axis_y = margin_top + plot_height
    draw.line((axis_x, margin_top, axis_x, axis_y), fill="#4b5563", width=2)
    draw.line((axis_x, axis_y, width - margin_right, axis_y), fill="#4b5563", width=2)
    for row_index, row in enumerate(rows):
        count = int(row["count"])
        bar_height = int(plot_height * count / max_count)
        x0 = margin_left + int(row_index * bar_width) + 2
        x1 = margin_left + int((row_index + 1) * bar_width) - 2
        y0 = axis_y - bar_height
        draw.rectangle((x0, y0, x1, axis_y), fill="#4f83cc", outline="#2f5f9c")
        if row_index % 2 == 0:
            label = f"{row['bin_left']:.2f}"
            draw.text((x0 - 4, axis_y + 10), label, fill="#172033", font=tick_font)
    draw.text((width // 2 - 110, height - 45), "Probability bin", fill="#172033", font=axis_font)
    draw.text((20, margin_top + plot_height // 2), "Count", fill="#172033", font=axis_font)
    image.save(path)


def plot_ecg_examples(
    path: Path,
    *,
    X: np.ndarray,
    case_rows: list[dict[str, object]],
    title: str,
) -> None:
    row_height = 180
    width = 1150
    height = 85 + row_height * len(case_rows)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24, bold=True)
    text_font = load_font(13)
    draw.text((25, 22), title, fill="#172033", font=title_font)
    for row_position, case in enumerate(case_rows):
        top = 75 + row_position * row_height
        left = 65
        right = width - 35
        bottom = top + row_height - 45
        signal = X[int(case["dataset_index"])]
        signal = signal.astype(float)
        low = float(np.quantile(signal, 0.01))
        high = float(np.quantile(signal, 0.99))
        if high <= low:
            high = low + 1.0
        clipped = np.clip(signal, low, high)
        x_coords = np.linspace(left, right, num=clipped.size)
        y_coords = bottom - ((clipped - low) / (high - low)) * (bottom - top)
        points = list(zip(x_coords.astype(int), y_coords.astype(int)))
        draw.rectangle((left, top, right, bottom), outline="#cbd5e1", width=1)
        draw.line(points, fill="#0f766e", width=1)
        label = (
            f"rank {case['rank']} | p(CurrentECG)={case['p_currentecg']:.4f} | "
            f"top alternative={case['top_non_current_label']} "
            f"({case['p_top_non_current']:.4f}) | margin={case['confidence_margin']:.4f}"
        )
        draw.text((left, bottom + 8), label, fill="#172033", font=text_font)
    image.save(path)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.model_path, device)
    dataset = np.load(args.dataset_path, allow_pickle=True)
    X = dataset["X"]
    rhythm_features = dataset["rhythm_features"].astype(np.float32)
    y = dataset["y"]
    labels = dataset["labels"]
    occurrence_ids = dataset["occurrence_ids"]
    device_ids = dataset["device_ids"]
    rhythm_mean = np.asarray(checkpoint["rhythm_feature_mean"], dtype=np.float32)
    rhythm_std = np.asarray(checkpoint["rhythm_feature_std"], dtype=np.float32)
    rhythm_features = (rhythm_features - rhythm_mean) / rhythm_std
    model = build_model(checkpoint, rhythm_features.shape[1], device)
    selected_indices = select_subset(
        subset=args.subset,
        y=y,
        labels=labels,
        device_ids=device_ids,
        metrics=metrics,
    )
    probabilities = predict_probabilities(
        model=model,
        X=X,
        rhythm_features=rhythm_features,
        indices=selected_indices,
        batch_size=args.batch_size,
        device=device,
    )
    current_label_index = int(np.where(labels == "CurrentECG")[0][0])
    p_current = probabilities[:, current_label_index]
    non_current_indices = [index for index in range(len(labels)) if index != current_label_index]
    non_current_probabilities = probabilities[:, non_current_indices]
    top_non_current_positions = np.argmax(non_current_probabilities, axis=1)
    top_non_current_indices = np.asarray(non_current_indices)[top_non_current_positions]
    top_non_current_probabilities = non_current_probabilities[
        np.arange(non_current_probabilities.shape[0]),
        top_non_current_positions,
    ]
    predicted_indices = np.argmax(probabilities, axis=1)
    margins = p_current - top_non_current_probabilities
    entropies = entropy(probabilities)

    prediction_counts = Counter(labels[index] for index in predicted_indices)
    alternative_counts = Counter(labels[index] for index in top_non_current_indices)
    summary = {
        "dataset_path": str(args.dataset_path),
        "model_path": str(args.model_path),
        "subset": args.subset,
        "analyzed_records": int(selected_indices.size),
        "labels": labels.tolist(),
        "official_prediction_counts": {
            str(label): int(prediction_counts.get(label, 0)) for label in labels
        },
        "top_non_current_counts": {
            str(label): int(alternative_counts.get(label, 0))
            for label in labels
            if str(label) != "CurrentECG"
        },
        "p_currentecg_summary": probability_summary(p_current),
        "top_non_current_probability_summary": probability_summary(
            top_non_current_probabilities
        ),
        "confidence_margin_summary": probability_summary(margins),
        "entropy_summary": probability_summary(entropies),
        "low_confidence_counts": {
            f"p_currentecg_below_{threshold}": int(np.sum(p_current < threshold))
            for threshold in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
        },
        "low_margin_counts": {
            f"margin_below_{threshold}": int(np.sum(margins < threshold))
            for threshold in (0.05, 0.10, 0.20, 0.30, 0.50)
        },
    }

    low_confidence_order = np.argsort(p_current)
    high_alternative_order = np.argsort(-top_non_current_probabilities)
    fieldnames = [
        "rank",
        "dataset_index",
        "occurrence_hash",
        "device_hash",
        "predicted_label",
        "top_non_current_label",
        "p_currentecg",
        "p_top_non_current",
        "confidence_margin",
        "entropy",
        *[f"p_{label}" for label in labels],
    ]
    low_confidence_rows = build_case_rows(
        indices=selected_indices,
        probabilities=probabilities,
        labels=labels,
        occurrence_ids=occurrence_ids,
        device_ids=device_ids,
        current_label_index=current_label_index,
        order=low_confidence_order,
        top_n=args.top_n,
    )
    high_alternative_rows = build_case_rows(
        indices=selected_indices,
        probabilities=probabilities,
        labels=labels,
        occurrence_ids=occurrence_ids,
        device_ids=device_ids,
        current_label_index=current_label_index,
        order=high_alternative_order,
        top_n=args.top_n,
    )
    write_csv(output_dir / "lowest_currentecg_confidence_cases.csv", low_confidence_rows, fieldnames)
    write_csv(output_dir / "highest_non_current_probability_cases.csv", high_alternative_rows, fieldnames)

    per_alternative_rows = []
    for label_index in non_current_indices:
        label = str(labels[label_index])
        order = np.argsort(-probabilities[:, label_index])
        rows = build_case_rows(
            indices=selected_indices,
            probabilities=probabilities,
            labels=labels,
            occurrence_ids=occurrence_ids,
            device_ids=device_ids,
            current_label_index=current_label_index,
            order=order,
            top_n=args.top_n,
        )
        for row in rows:
            row["alternative_ranked_for"] = label
        per_alternative_rows.extend(rows)
    write_csv(
        output_dir / "highest_probability_by_alternative_label_cases.csv",
        per_alternative_rows,
        ["alternative_ranked_for", *fieldnames],
    )

    current_histogram_rows = write_histogram_csv(
        output_dir / "currentecg_probability_histogram.csv",
        p_current,
    )
    alternative_histogram_rows = write_histogram_csv(
        output_dir / "top_non_current_probability_histogram.csv",
        top_non_current_probabilities,
    )
    plot_histogram(
        output_dir / "currentecg_probability_histogram.png",
        current_histogram_rows,
        title="CurrentECG Confidence Distribution",
    )
    plot_histogram(
        output_dir / "top_non_current_probability_histogram.png",
        alternative_histogram_rows,
        title="Top Non-CurrentECG Probability Distribution",
    )
    plot_ecg_examples(
        output_dir / "lowest_currentecg_confidence_examples.png",
        X=X,
        case_rows=low_confidence_rows[: min(args.plot_examples, len(low_confidence_rows))],
        title="Lowest CurrentECG Confidence Examples",
    )
    plot_ecg_examples(
        output_dir / "highest_non_current_probability_examples.png",
        X=X,
        case_rows=high_alternative_rows[: min(args.plot_examples, len(high_alternative_rows))],
        title="Highest Alternative-Label Probability Examples",
    )
    summary["outputs"] = {
        "summary_json": str(output_dir / "summary.json"),
        "lowest_currentecg_confidence_cases": str(
            output_dir / "lowest_currentecg_confidence_cases.csv"
        ),
        "highest_non_current_probability_cases": str(
            output_dir / "highest_non_current_probability_cases.csv"
        ),
        "highest_probability_by_alternative_label_cases": str(
            output_dir / "highest_probability_by_alternative_label_cases.csv"
        ),
        "currentecg_probability_histogram": str(
            output_dir / "currentecg_probability_histogram.png"
        ),
        "top_non_current_probability_histogram": str(
            output_dir / "top_non_current_probability_histogram.png"
        ),
        "lowest_currentecg_confidence_examples": str(
            output_dir / "lowest_currentecg_confidence_examples.png"
        ),
        "highest_non_current_probability_examples": str(
            output_dir / "highest_non_current_probability_examples.png"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze confidence of CurrentECG predictions without retraining."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--subset",
        choices=(
            "test_currentecg",
            "validation_currentecg",
            "train_currentecg",
            "all_currentecg",
        ),
        default="test_currentecg",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--plot-examples", type=int, default=8)
    return parser


def main() -> int:
    summary = analyze(build_parser().parse_args())
    print(f"Subset: {summary['subset']}")
    print(f"Analyzed records: {summary['analyzed_records']}")
    print(f"Official prediction counts: {summary['official_prediction_counts']}")
    print(f"Top non-CurrentECG counts: {summary['top_non_current_counts']}")
    print(f"Lowest p(CurrentECG): {summary['p_currentecg_summary']['min']:.6f}")
    print(
        "Highest top non-CurrentECG probability: "
        f"{summary['top_non_current_probability_summary']['max']:.6f}"
    )
    print(f"Summary: {summary['outputs']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
