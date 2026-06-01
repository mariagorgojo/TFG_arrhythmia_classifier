from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".pythonlibs"

try:
    import numpy as np
except ModuleNotFoundError:
    if LOCAL_SITE_PACKAGES.exists():
        sys.path.append(str(LOCAL_SITE_PACKAGES))
    import numpy as np

if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(LOCAL_SITE_PACKAGES))
for dll_dir in (
    LOCAL_SITE_PACKAGES / "torch" / "lib",
    LOCAL_SITE_PACKAGES / "numpy.libs",
    LOCAL_SITE_PACKAGES / "scipy.libs",
):
    if dll_dir.exists():
        os.add_dll_directory(str(dll_dir))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "datasets" / "ecg_training_dataset_arrhythmia4_features_deidentified.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cnn1d_arrhythmia4_baseline"


class ECG1DCNN(nn.Module):
    """Small 1D CNN for fixed-length single-lead ECG windows."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
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
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_dataset(dataset_path: Path) -> dict[str, np.ndarray]:
    data = np.load(dataset_path)
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "labels": data["labels"].astype(str),
        "device_ids": data["device_ids"].astype(str),
    }


def split_by_device(
    X: np.ndarray,
    y: np.ndarray,
    device_ids: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    split_strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    if split_strategy == "stratified_group":
        n_splits = max(2, round(1.0 / test_size))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        full_distribution = np.bincount(y, minlength=int(np.max(y)) + 1) / len(y)
        best_score = float("inf")
        best_split: tuple[np.ndarray, np.ndarray] | None = None

        for train_index, test_index in splitter.split(X, y, groups=device_ids):
            test_distribution = (
                np.bincount(y[test_index], minlength=full_distribution.size)
                / test_index.size
            )
            size_penalty = abs((test_index.size / len(y)) - test_size)
            label_penalty = float(np.abs(test_distribution - full_distribution).sum())
            score = size_penalty + label_penalty
            if score < best_score:
                best_score = score
                best_split = (train_index, test_index)

        if best_split is None:
            raise RuntimeError("Could not create a stratified group split.")
        return best_split

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(splitter.split(X, y, groups=device_ids))
    return train_index, test_index


def compute_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X[indices], dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y[indices], dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
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

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
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
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            total_loss += float(loss.item()) * y_batch.size(0)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            true_labels.extend(y_batch.cpu().numpy().tolist())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = float(accuracy_score(true_labels, predictions))
    return mean_loss, accuracy, np.asarray(true_labels), np.asarray(predictions)


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray, labels: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + [f"predicted_{label}" for label in labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [int(value) for value in row])


def plot_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    title: str = "CNN 1D Confusion Matrix",
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = [
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()

    num_labels = len(labels)
    cell_size = 115
    left_margin = 190
    top_margin = 165
    right_margin = 45
    bottom_margin = 75
    width = left_margin + num_labels * cell_size + right_margin
    height = top_margin + num_labels * cell_size + bottom_margin

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    axis_font = load_font(17, bold=True)
    label_font = load_font(16)
    cell_font = load_font(20, bold=True)

    draw.text((30, 25), title, fill="#172033", font=title_font)
    draw.text(
        (left_margin + (num_labels * cell_size) // 2 - 65, 78),
        "Predicted label",
        fill="#172033",
        font=axis_font,
    )
    draw.text((30, top_margin + (num_labels * cell_size) // 2 - 12), "True label", fill="#172033", font=axis_font)

    max_value = int(matrix.max()) if matrix.size else 1
    max_value = max(max_value, 1)

    for column_index, label in enumerate(labels):
        x = left_margin + column_index * cell_size + cell_size // 2
        draw.text((x - 42, top_margin - 42), str(label), fill="#172033", font=label_font)

    for row_index, label in enumerate(labels):
        y = top_margin + row_index * cell_size + cell_size // 2
        draw.text((left_margin - 140, y - 10), str(label), fill="#172033", font=label_font)

    for row_index in range(num_labels):
        for column_index in range(num_labels):
            value = int(matrix[row_index, column_index])
            intensity = int(255 - 185 * (value / max_value))
            fill = (intensity, intensity + 12 if intensity < 243 else 255, 255)
            x0 = left_margin + column_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#6b7685", width=1)
            text_color = "white" if value > max_value * 0.55 else "#172033"
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=cell_font)
            draw.text(
                (
                    x0 + (cell_size - (bbox[2] - bbox[0])) // 2,
                    y0 + (cell_size - (bbox[3] - bbox[1])) // 2,
                ),
                text,
                fill=text_color,
                font=cell_font,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def train_and_evaluate(
    dataset_path: Path,
    output_dir: Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    patience: int = 5,
    split_strategy: str = "stratified_group",
    monitor_metric: str = "macro_f1",
) -> dict[str, object]:
    set_reproducible_seed(random_state)
    dataset = load_training_dataset(dataset_path)
    X = dataset["X"]
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
    train_loader = make_loader(X, y, train_index, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(X, y, test_index, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECG1DCNN(num_classes=num_classes).to(device)
    class_weights = compute_class_weights(y[train_index], num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "cnn1d_model.pt"
    history: list[dict[str, float | int]] = []
    best_monitor_value = -float("inf") if monitor_metric != "test_loss" else float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        test_loss, test_accuracy, epoch_y_true, epoch_y_pred = evaluate(
            model, test_loader, loss_fn, device
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
                    "random_state": random_state,
                    "architecture": "ECG1DCNN",
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
        model, test_loader, loss_fn, device
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
    plot_confusion_matrix(confusion_png_path, matrix, labels)

    train_counts = Counter(labels[label_index] for label_index in y[train_index])
    test_counts = Counter(labels[label_index] for label_index in y[test_index])
    class_weight_values = class_weights.detach().cpu().numpy()
    metrics = {
        "model_type": "ECG1DCNN",
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
        description="Train a 1D CNN on prepared ECG arrays."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Prepared training dataset path. Default: {DEFAULT_DATASET_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for model and metrics. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=("stratified_group", "group_shuffle"),
        default="stratified_group",
        help=(
            "How to create train/test sets. stratified_group keeps devices separated "
            "while trying to preserve class proportions."
        ),
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
        help=(
            "Metric used to save the best checkpoint. macro_f1 is recommended "
            "for imbalanced multiclass ECG datasets."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_and_evaluate(
        args.dataset_path,
        args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        split_strategy=args.split_strategy,
        monitor_metric=args.monitor_metric,
    )
    print(f"Model: {metrics['model_path']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test loss: {metrics['test_loss']:.4f}")
    print(f"Train examples: {metrics['train_examples']}")
    print(f"Test examples: {metrics['test_examples']}")
    print(f"Train devices: {metrics['train_device_count']}")
    print(f"Test devices: {metrics['test_device_count']}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
