from __future__ import annotations

import argparse
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from arrhythmia_classifier.cnn1d_classifier import (
    compute_class_weights,
    plot_confusion_matrix,
    set_reproducible_seed,
    split_by_device,
    write_confusion_matrix_csv,
)
from arrhythmia_classifier.hybrid_cnn1d_classifier import (
    ECGHybridCNN,
    evaluate,
    load_training_dataset,
    make_loader,
    standardize_rhythm_features,
    train_one_epoch,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "datasets" / "ecg_training_dataset_arrhythmia4_features_deidentified.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid_cnn1d_arrhythmia4_train_val_test"


def split_train_validation_test_by_device(
    X: np.ndarray,
    y: np.ndarray,
    device_ids: np.ndarray,
    *,
    validation_size: float,
    test_size: float,
    random_state: int,
    split_strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create device-disjoint train, validation and test indices."""
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be greater than zero.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be lower than one.")

    development_index, test_index = split_by_device(
        X,
        y,
        device_ids,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train_relative_index, validation_relative_index = split_by_device(
        X[development_index],
        y[development_index],
        device_ids[development_index],
        test_size=relative_validation_size,
        random_state=random_state + 1,
        split_strategy=split_strategy,
    )
    train_index = development_index[train_relative_index]
    validation_index = development_index[validation_relative_index]

    train_devices = set(device_ids[train_index])
    validation_devices = set(device_ids[validation_index])
    test_devices = set(device_ids[test_index])
    if train_devices & validation_devices:
        raise RuntimeError("A device group appears in both train and validation.")
    if train_devices & test_devices:
        raise RuntimeError("A device group appears in both train and test.")
    if validation_devices & test_devices:
        raise RuntimeError("A device group appears in both validation and test.")

    return train_index, validation_index, test_index


def _report_outputs(
    output_dir: Path,
    *,
    prefix: str,
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, object], dict[str, str]]:
    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(labels)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

    report_path = output_dir / f"{prefix}_classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    confusion_csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    write_confusion_matrix_csv(confusion_csv_path, matrix, labels)
    confusion_png_path = output_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion_matrix(confusion_png_path, matrix, labels, title=title)
    return report_dict, {
        "classification_report": str(report_path),
        "confusion_matrix_csv": str(confusion_csv_path),
        "confusion_matrix_png": str(confusion_png_path),
    }


def train_and_evaluate(
    dataset_path: Path,
    output_dir: Path,
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    split_strategy: str = "stratified_group",
    random_state: int = 42,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    patience: int = 5,
    monitor_metric: str = "macro_f1",
) -> dict[str, object]:
    set_reproducible_seed(random_state)
    dataset = load_training_dataset(dataset_path)
    X = dataset["X"]
    rhythm_features = dataset["rhythm_features"]
    rhythm_feature_names = dataset["rhythm_feature_names"]
    y = dataset["y"]
    labels = dataset["labels"]
    device_ids = dataset["device_ids"]
    num_classes = len(labels)

    train_index, validation_index, test_index = split_train_validation_test_by_device(
        X,
        y,
        device_ids,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )
    rhythm_features, rhythm_mean, rhythm_std = standardize_rhythm_features(
        rhythm_features,
        train_index,
    )

    train_loader = make_loader(
        X, rhythm_features, y, train_index, batch_size=batch_size, shuffle=True
    )
    validation_loader = make_loader(
        X, rhythm_features, y, validation_index, batch_size=batch_size, shuffle=False
    )
    test_loader = make_loader(
        X, rhythm_features, y, test_index, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGHybridCNN(
        num_classes=num_classes,
        num_rhythm_features=rhythm_features.shape[1],
    ).to(device)
    class_weights = compute_class_weights(y[train_index], num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "hybrid_cnn1d_train_val_test_model.pt"
    history: list[dict[str, float | int]] = []
    best_monitor_value = -float("inf") if monitor_metric != "validation_loss" else float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        validation_loss, validation_accuracy, validation_y_true, validation_y_pred = evaluate(
            model, validation_loader, loss_fn, device
        )
        validation_macro_f1 = float(
            f1_score(validation_y_true, validation_y_pred, average="macro", zero_division=0)
        )
        validation_weighted_f1 = float(
            f1_score(validation_y_true, validation_y_pred, average="weighted", zero_division=0)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
                "validation_weighted_f1": validation_weighted_f1,
            }
        )
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_acc={validation_accuracy:.4f}, "
            f"validation_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )

        current_monitor_value = {
            "validation_loss": validation_loss,
            "macro_f1": validation_macro_f1,
            "weighted_f1": validation_weighted_f1,
        }[monitor_metric]
        improved = (
            current_monitor_value < best_monitor_value
            if monitor_metric == "validation_loss"
            else current_monitor_value > best_monitor_value
        )

        if improved:
            best_monitor_value = current_monitor_value
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "labels": labels.tolist(),
                    "dataset_path": str(dataset_path),
                    "target_samples": int(X.shape[1]),
                    "rhythm_feature_names": rhythm_feature_names.tolist(),
                    "rhythm_feature_mean": rhythm_mean.tolist(),
                    "rhythm_feature_std": rhythm_std.tolist(),
                    "random_state": random_state,
                    "architecture": "ECGHybridCNN",
                    "evaluation_protocol": "device_disjoint_train_validation_test",
                    "monitor_metric": monitor_metric,
                    "monitor_value": best_monitor_value,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping: {monitor_metric} did not improve on validation.",
                    flush=True,
                )
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_loss, validation_accuracy, validation_y_true, validation_y_pred = evaluate(
        model, validation_loader, loss_fn, device
    )
    test_loss, test_accuracy, test_y_true, test_y_pred = evaluate(
        model, test_loader, loss_fn, device
    )

    validation_report, validation_outputs = _report_outputs(
        output_dir,
        prefix="validation",
        title="Hybrid CNN 1D Validation Confusion Matrix",
        y_true=validation_y_true,
        y_pred=validation_y_pred,
        labels=labels,
    )
    test_report, test_outputs = _report_outputs(
        output_dir,
        prefix="test",
        title="Hybrid CNN 1D Final Test Confusion Matrix",
        y_true=test_y_true,
        y_pred=test_y_pred,
        labels=labels,
    )

    class_weight_values = class_weights.detach().cpu().numpy()
    metrics = {
        "model_type": "ECGHybridCNN",
        "evaluation_protocol": "device_disjoint_train_validation_test",
        "methodological_note": (
            "Training updates use only train groups. The best checkpoint is selected "
            "using validation groups. Test groups are evaluated once after training."
        ),
        "dataset_path": str(dataset_path),
        "model_path": str(best_model_path),
        "validation_size_requested": validation_size,
        "test_size_requested": test_size,
        "split_strategy": split_strategy,
        "random_state": random_state,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "monitor_metric": monitor_metric,
        "best_validation_monitor_value": best_monitor_value,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_samples": int(X.shape[1]),
        "rhythm_feature_names": rhythm_feature_names.tolist(),
        "train_examples": int(train_index.size),
        "validation_examples": int(validation_index.size),
        "test_examples": int(test_index.size),
        "train_device_count": int(np.unique(device_ids[train_index]).size),
        "validation_device_count": int(np.unique(device_ids[validation_index]).size),
        "test_device_count": int(np.unique(device_ids[test_index]).size),
        "train_label_counts": dict(Counter(labels[y[train_index]])),
        "validation_label_counts": dict(Counter(labels[y[validation_index]])),
        "test_label_counts": dict(Counter(labels[y[test_index]])),
        "class_weights_from_train_only": {
            str(label): float(weight)
            for label, weight in zip(labels, class_weight_values)
        },
        "history": history,
        "validation": {
            "accuracy": validation_accuracy,
            "loss": validation_loss,
            "classification_report": validation_report,
            "outputs": validation_outputs,
        },
        "test": {
            "accuracy": test_accuracy,
            "loss": test_loss,
            "classification_report": test_report,
            "outputs": test_outputs,
        },
        "device": str(device),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a hybrid ECG CNN with device-disjoint train, validation and test sets."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=("stratified_group", "group_shuffle"),
        default="stratified_group",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--monitor-metric",
        choices=("macro_f1", "weighted_f1", "validation_loss"),
        default="macro_f1",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_and_evaluate(
        args.dataset_path,
        args.output_dir,
        validation_size=args.validation_size,
        test_size=args.test_size,
        split_strategy=args.split_strategy,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
    )
    print(f"Model: {metrics['model_path']}")
    print(f"Validation accuracy: {metrics['validation']['accuracy']:.4f}")
    print(f"Final test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Train examples: {metrics['train_examples']}")
    print(f"Validation examples: {metrics['validation_examples']}")
    print(f"Test examples: {metrics['test_examples']}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
