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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from arrhythmia_classifier.cnn1d_classifier import (
    compute_class_weights,
    plot_confusion_matrix,
    set_reproducible_seed,
    split_by_device,
    write_confusion_matrix_csv,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "datasets" / "ecg_training_dataset_arrhythmia4_features_deidentified.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid_cnn1d_arrhythmia4"


class ECGHybridCNN(nn.Module):
    """CNN for ECG morphology combined with rhythm/marker features."""

    def __init__(self, num_classes: int, num_rhythm_features: int) -> None:
        super().__init__()
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.rhythm_branch = nn.Sequential(
            nn.Linear(num_rhythm_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(128 + 32, num_classes),
        )

    def forward(self, ecg: torch.Tensor, rhythm_features: torch.Tensor) -> torch.Tensor:
        ecg_embedding = self.ecg_branch(ecg)
        rhythm_embedding = self.rhythm_branch(rhythm_features)
        combined = torch.cat([ecg_embedding, rhythm_embedding], dim=1)
        return self.classifier(combined)


def load_training_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    data = np.load(dataset_path)
    if "rhythm_features" not in data:
        raise KeyError(
            "The dataset does not contain rhythm_features. Rebuild it with the "
            "current dataset_loader.py before training the hybrid model."
        )
    return {
        "X": data["X"].astype(np.float32),
        "rhythm_features": data["rhythm_features"].astype(np.float32),
        "rhythm_feature_names": data["rhythm_feature_names"].astype(str),
        "y": data["y"].astype(np.int64),
        "labels": data["labels"].astype(str),
        "device_ids": data["device_ids"].astype(str),
    }


def standardize_rhythm_features(
    features: np.ndarray,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_features = features[train_index]
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    standardized = (features - mean) / std
    return standardized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def make_loader(
    X: np.ndarray,
    rhythm_features: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X[indices], dtype=torch.float32).unsqueeze(1)
    rhythm_tensor = torch.tensor(rhythm_features[indices], dtype=torch.float32)
    y_tensor = torch.tensor(y[indices], dtype=torch.long)
    dataset = TensorDataset(X_tensor, rhythm_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    predictions: list[int] = []
    true_labels: list[int] = []

    for X_batch, rhythm_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        rhythm_batch = rhythm_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch, rhythm_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y_batch.size(0)
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        true_labels.extend(y_batch.detach().cpu().numpy().tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = float(accuracy_score(true_labels, predictions))
    return mean_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    true_labels: list[int] = []

    with torch.no_grad():
        for X_batch, rhythm_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            rhythm_batch = rhythm_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch, rhythm_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += float(loss.item()) * y_batch.size(0)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            true_labels.extend(y_batch.cpu().numpy().tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = float(accuracy_score(true_labels, predictions))
    return mean_loss, accuracy, np.asarray(true_labels), np.asarray(predictions)


def train_and_evaluate(
    dataset_path: Path,
    output_dir: Path,
    *,
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

    train_index, test_index = split_by_device(
        X,
        y,
        device_ids,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )
    rhythm_features, rhythm_mean, rhythm_std = standardize_rhythm_features(
        rhythm_features,
        train_index,
    )

    train_loader = make_loader(
        X,
        rhythm_features,
        y,
        train_index,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = make_loader(
        X,
        rhythm_features,
        y,
        test_index,
        batch_size=batch_size,
        shuffle=False,
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
    best_model_path = output_dir / "hybrid_cnn1d_model.pt"
    history: list[dict[str, float | int]] = []
    best_monitor_value = -float("inf") if monitor_metric != "test_loss" else float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )
        test_loss, test_accuracy, epoch_y_true, epoch_y_pred = evaluate(
            model,
            test_loader,
            loss_fn,
            device,
        )
        epoch_macro_f1 = float(
            f1_score(epoch_y_true, epoch_y_pred, average="macro", zero_division=0)
        )
        epoch_weighted_f1 = float(
            f1_score(epoch_y_true, epoch_y_pred, average="weighted", zero_division=0)
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "test_macro_f1": epoch_macro_f1,
                "test_weighted_f1": epoch_weighted_f1,
            }
        )
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_accuracy:.4f}, "
            f"macro_f1={epoch_macro_f1:.4f}",
            flush=True,
        )

        current_monitor_value = {
            "test_loss": test_loss,
            "macro_f1": epoch_macro_f1,
            "weighted_f1": epoch_weighted_f1,
        }[monitor_metric]
        improved = (
            current_monitor_value < best_monitor_value
            if monitor_metric == "test_loss"
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
                    "monitor_metric": monitor_metric,
                    "monitor_value": best_monitor_value,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping: {monitor_metric} did not improve.",
                    flush=True,
                )
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_loss, final_accuracy, y_true, y_pred = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=labels,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    report_path = output_dir / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    confusion_csv_path = output_dir / "confusion_matrix.csv"
    write_confusion_matrix_csv(confusion_csv_path, matrix, labels)
    confusion_png_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        confusion_png_path,
        matrix,
        labels,
        title="Hybrid CNN 1D Confusion Matrix",
    )

    train_counts = Counter(labels[label_index] for label_index in y[train_index])
    test_counts = Counter(labels[label_index] for label_index in y[test_index])
    class_weight_values = class_weights.detach().cpu().numpy()
    metrics = {
        "model_type": "ECGHybridCNN",
        "dataset_path": str(dataset_path),
        "model_path": str(best_model_path),
        "accuracy": final_accuracy,
        "test_loss": final_loss,
        "test_size": test_size,
        "split_strategy": split_strategy,
        "random_state": random_state,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "monitor_metric": monitor_metric,
        "best_monitor_value": best_monitor_value,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "target_samples": int(X.shape[1]),
        "rhythm_feature_names": rhythm_feature_names.tolist(),
        "train_examples": int(train_index.size),
        "test_examples": int(test_index.size),
        "train_device_count": int(np.unique(device_ids[train_index]).size),
        "test_device_count": int(np.unique(device_ids[test_index]).size),
        "labels": labels.tolist(),
        "train_label_counts": dict(train_counts),
        "test_label_counts": dict(test_counts),
        "class_weights": {
            str(label): float(weight)
            for label, weight in zip(labels, class_weight_values)
        },
        "history": history,
        "classification_report": report_dict,
        "outputs": {
            "classification_report": str(report_path),
            "confusion_matrix_csv": str(confusion_csv_path),
            "confusion_matrix_png": str(confusion_png_path),
        },
        "device": str(device),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a hybrid 1D CNN using ECG signal plus rhythm features."
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        choices=("macro_f1", "weighted_f1", "test_loss"),
        default="macro_f1",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_and_evaluate(
        args.dataset_path,
        args.output_dir,
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
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Train examples: {metrics['train_examples']}")
    print(f"Test examples: {metrics['test_examples']}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
