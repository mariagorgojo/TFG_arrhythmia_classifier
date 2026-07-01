"""Frozen MOMENT foundation-model comparison for the five-class ECG dataset.

This script is meant to be run in Google Colab after installing MOMENT. It uses
the same device-disjoint train/validation/test split as the custom CNN models,
loads a pretrained MOMENT time-series foundation model, replaces its final
classification layer with a five-class head, freezes the pretrained network and
trains only that new final classification layer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from arrhythmia_classifier.cnn1d_classifier import (
    compute_class_weights,
    plot_confusion_matrix,
    set_reproducible_seed,
    write_confusion_matrix_csv,
)
from arrhythmia_classifier.hybrid_cnn1d_classifier import load_training_dataset
from arrhythmia_classifier.hybrid_cnn1d_train_val_test import (
    split_train_validation_test_by_device,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "training_dataset"
    / "ecg_training_dataset_currentecg5_features.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "foundation_moment_currentecg5"
DEFAULT_MODEL_NAME = "AutonLab/MOMENT-1-small"
DEFAULT_MOMENT_CONTEXT_LENGTH = 512


@dataclass(frozen=True)
class MomentRunConfig:
    model_name: str
    context_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    random_state: int
    validation_size: float
    test_size: float
    split_strategy: str
    progress_every: int


def resize_waveforms_for_moment(
    X: np.ndarray,
    *,
    context_length: int,
) -> tuple[np.ndarray, str]:
    """Convert project ECG windows to MOMENT context length.

    The final dataset stores 2,048 samples per ECG. MOMENT tutorials use a
    512-timestep context, so the default conversion is deterministic average
    pooling over groups of four samples. If a future dataset length is not an
    exact multiple, linear interpolation is used instead and recorded in the
    metrics file.
    """

    if X.shape[1] == context_length:
        return X.astype(np.float32, copy=False), "none"

    if X.shape[1] % context_length == 0:
        factor = X.shape[1] // context_length
        pooled = X.reshape(X.shape[0], context_length, factor).mean(axis=2)
        return pooled.astype(np.float32), f"average_pool_factor_{factor}"

    source_grid = np.linspace(0.0, 1.0, X.shape[1], dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, context_length, dtype=np.float32)
    resized = np.empty((X.shape[0], context_length), dtype=np.float32)
    for row_index, row in enumerate(X):
        resized[row_index] = np.interp(target_grid, source_grid, row).astype(np.float32)
    return resized, "linear_interpolation"


def make_waveform_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X[indices], dtype=torch.float32).unsqueeze(1),
        torch.tensor(y[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _replace_last_linear(module: nn.Module, num_classes: int) -> tuple[str, nn.Linear]:
    """Replace the last Linear layer inside a module and return its local path."""

    last_parent: nn.Module | None = None
    last_name: str | None = None
    last_path: str | None = None
    last_linear: nn.Linear | None = None

    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            if "." in name:
                parent_path, child_name = name.rsplit(".", 1)
                parent = module.get_submodule(parent_path)
            else:
                parent = module
                child_name = name
            last_parent = parent
            last_name = child_name
            last_path = name
            last_linear = child

    if last_parent is None or last_name is None or last_path is None or last_linear is None:
        raise RuntimeError("Could not find a Linear layer to replace in MOMENT head.")

    new_linear = nn.Linear(last_linear.in_features, num_classes)
    setattr(last_parent, last_name, new_linear)
    return last_path, new_linear


def build_frozen_moment_classifier(
    *,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, str, int, int]:
    """Load MOMENT, replace its final layer and freeze all other parameters."""

    try:
        from momentfm import MOMENTPipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MOMENT is not installed. In Colab run: "
            "python -m pip install git+https://github.com/"
            "moment-timeseries-foundation-model/moment.git"
        ) from exc

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={
            "task_name": "classification",
            "n_channels": 1,
            "num_class": num_classes,
        },
    )
    model.init()

    if not hasattr(model, "head"):
        raise RuntimeError("Expected MOMENT classification pipeline to expose model.head.")

    head = model.head
    replaced_layer = "head."
    if hasattr(head, "linear") and isinstance(head.linear, nn.Linear):
        in_features = head.linear.in_features
        new_classifier = nn.Linear(in_features, num_classes)
        head.linear = new_classifier
        replaced_layer += "linear"
    else:
        local_layer_path, new_classifier = _replace_last_linear(head, num_classes)
        replaced_layer += local_layer_path

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in new_classifier.parameters():
        parameter.requires_grad = True

    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return model.to(device), replaced_layer, trainable_parameters, frozen_parameters


def _logits_from_output(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if logits is None:
        raise RuntimeError("MOMENT output did not contain logits.")
    return logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    progress_every: int = 0,
) -> tuple[float, float]:
    # Linear probing keeps the frozen encoder deterministic while the new
    # classification layer is optimized.
    model.eval()
    if hasattr(model, "head"):
        model.head.train()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch_index, (X_batch, y_batch) in enumerate(loader, start=1):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _logits_from_output(model(x_enc=X_batch))
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y_batch.size(0)
        y_true.extend(y_batch.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        if progress_every and batch_index % progress_every == 0:
            print(
                f"  train batch {batch_index}/{len(loader)} "
                f"loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / len(loader.dataset), float(accuracy_score(y_true, y_pred))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    progress_every: int = 0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch_index, (X_batch, y_batch) in enumerate(loader, start=1):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = _logits_from_output(model(x_enc=X_batch))
            loss = loss_fn(logits, y_batch)
            total_loss += float(loss.item()) * y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            if progress_every and batch_index % progress_every == 0:
                print(
                    f"  eval batch {batch_index}/{len(loader)}",
                    flush=True,
                )

    return (
        total_loss / len(loader.dataset),
        float(accuracy_score(y_true, y_pred)),
        np.asarray(y_true),
        np.asarray(y_pred),
    )


def arrhythmia_only_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
) -> float:
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    arrhythmia_indices = [
        index for index, label in enumerate(labels) if str(label) != "CurrentECG"
    ]
    return float(np.mean([per_class_f1[index] for index in arrhythmia_indices]))


def write_report_outputs(
    *,
    output_dir: Path,
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
    matrix_csv_path = output_dir / f"{prefix}_confusion_matrix.csv"
    write_confusion_matrix_csv(matrix_csv_path, matrix, labels)
    matrix_png_path = output_dir / f"{prefix}_confusion_matrix.png"
    plot_confusion_matrix(matrix_png_path, matrix, labels, title=title)
    return report_dict, {
        "classification_report": str(report_path),
        "confusion_matrix_csv": str(matrix_csv_path),
        "confusion_matrix_png": str(matrix_png_path),
    }


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    *,
    config: MomentRunConfig,
) -> dict[str, object]:
    set_reproducible_seed(config.random_state)
    dataset = load_training_dataset(dataset_path)
    X_original = dataset["X"]
    y = dataset["y"]
    labels = dataset["labels"]
    device_ids = dataset["device_ids"]

    train_index, validation_index, test_index = split_train_validation_test_by_device(
        X_original,
        y,
        device_ids,
        validation_size=config.validation_size,
        test_size=config.test_size,
        random_state=config.random_state,
        split_strategy=config.split_strategy,
    )

    X, resize_method = resize_waveforms_for_moment(
        X_original,
        context_length=config.context_length,
    )

    train_loader = make_waveform_loader(
        X, y, train_index, batch_size=config.batch_size, shuffle=True
    )
    validation_loader = make_waveform_loader(
        X, y, validation_index, batch_size=config.batch_size, shuffle=False
    )
    test_loader = make_waveform_loader(
        X, y, test_index, batch_size=config.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, replaced_layer, trainable_parameters, frozen_parameters = (
        build_frozen_moment_classifier(
            model_name=config.model_name,
            num_classes=len(labels),
            device=device,
        )
    )
    class_weights = compute_class_weights(y[train_index], len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_macro_f1 = -float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            progress_every=config.progress_every,
        )
        validation_loss, validation_accuracy, validation_true, validation_pred = evaluate(
            model,
            validation_loader,
            loss_fn,
            device,
            progress_every=config.progress_every,
        )
        validation_macro_f1 = float(
            f1_score(validation_true, validation_pred, average="macro", zero_division=0)
        )
        validation_weighted_f1 = float(
            f1_score(
                validation_true,
                validation_pred,
                average="weighted",
                zero_division=0,
            )
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
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, "
            f"validation_loss={validation_loss:.4f}, "
            f"validation_acc={validation_accuracy:.4f}, "
            f"validation_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )

        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping: validation macro F1 did not improve.", flush=True)
                break

    if best_state is not None:
        model.head.load_state_dict(best_state)

    validation_loss, validation_accuracy, validation_true, validation_pred = evaluate(
        model,
        validation_loader,
        loss_fn,
        device,
        progress_every=config.progress_every,
    )
    test_loss, test_accuracy, test_true, test_pred = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
        progress_every=config.progress_every,
    )

    validation_report, validation_outputs = write_report_outputs(
        output_dir=output_dir,
        prefix="validation",
        title="MOMENT Foundation Model Validation Confusion Matrix",
        y_true=validation_true,
        y_pred=validation_pred,
        labels=labels,
    )
    test_report, test_outputs = write_report_outputs(
        output_dir=output_dir,
        prefix="test",
        title="MOMENT Foundation Model Final Test Confusion Matrix",
        y_true=test_true,
        y_pred=test_pred,
        labels=labels,
    )

    head_path = output_dir / "foundation_moment_currentecg5_head.pt"
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "model_name": config.model_name,
            "labels": labels.tolist(),
            "dataset_path": str(dataset_path),
            "original_target_samples": int(X_original.shape[1]),
            "moment_context_length": config.context_length,
            "resize_method": resize_method,
            "replaced_layer": replaced_layer,
            "training_protocol": "frozen_moment_encoder_train_new_classification_head",
            "config": asdict(config),
        },
        head_path,
    )

    class_weight_values = class_weights.detach().cpu().numpy()
    label_counts = lambda idx: dict(Counter(labels[y[idx]]))
    metrics = {
        "model_type": "FrozenMOMENTFoundationClassifier",
        "model_name": config.model_name,
        "dataset_path": str(dataset_path),
        "head_path": str(head_path),
        "training_protocol": "frozen_moment_encoder_train_new_classification_head",
        "last_layer_replacement": replaced_layer,
        "trainable_parameters": int(trainable_parameters),
        "frozen_parameters": int(frozen_parameters),
        "original_target_samples": int(X_original.shape[1]),
        "moment_context_length": config.context_length,
        "resize_method": resize_method,
        "evaluation_protocol": "device_disjoint_train_validation_test",
        "selection_metric": "validation_macro_f1",
        "validation_size_requested": config.validation_size,
        "test_size_requested": config.test_size,
        "split_strategy": config.split_strategy,
        "random_state": config.random_state,
        "labels": labels.tolist(),
        "train_examples": int(train_index.size),
        "validation_examples": int(validation_index.size),
        "test_examples": int(test_index.size),
        "train_device_count": int(np.unique(device_ids[train_index]).size),
        "validation_device_count": int(np.unique(device_ids[validation_index]).size),
        "test_device_count": int(np.unique(device_ids[test_index]).size),
        "train_label_counts": label_counts(train_index),
        "validation_label_counts": label_counts(validation_index),
        "test_label_counts": label_counts(test_index),
        "class_weights_from_train_only": {
            str(label): float(weight)
            for label, weight in zip(labels, class_weight_values)
        },
        "config": asdict(config),
        "history": history,
        "validation": {
            "loss": validation_loss,
            "accuracy": validation_accuracy,
            "macro_f1": float(
                f1_score(
                    validation_true,
                    validation_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    validation_true,
                    validation_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "arrhythmia_only_macro_f1": arrhythmia_only_macro_f1(
                validation_true,
                validation_pred,
                labels,
            ),
            "classification_report": validation_report,
            "outputs": validation_outputs,
        },
        "test": {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "macro_f1": float(
                f1_score(test_true, test_pred, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(test_true, test_pred, average="weighted", zero_division=0)
            ),
            "arrhythmia_only_macro_f1": arrhythmia_only_macro_f1(
                test_true,
                test_pred,
                labels,
            ),
            "classification_report": test_report,
            "outputs": test_outputs,
        },
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a frozen MOMENT foundation-model classification head on the "
            "five-class CurrentECG dataset."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--context-length", type=int, default=DEFAULT_MOMENT_CONTEXT_LENGTH)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--split-strategy",
        choices=("stratified_group", "group_shuffle"),
        default="stratified_group",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N batches. Use 0 to disable batch progress.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = MomentRunConfig(
        model_name=args.model_name,
        context_length=args.context_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        random_state=args.random_state,
        validation_size=args.validation_size,
        test_size=args.test_size,
        split_strategy=args.split_strategy,
        progress_every=args.progress_every,
    )
    metrics = run_experiment(args.dataset_path, args.output_dir, config=config)
    print("\n=== MOMENT foundation-model comparison ===")
    print(f"Model: {metrics['model_name']}")
    print(f"Layer replaced: {metrics['last_layer_replacement']}")
    print(f"Trainable parameters: {metrics['trainable_parameters']}")
    print(f"Frozen parameters: {metrics['frozen_parameters']}")
    print(f"Validation macro F1: {metrics['validation']['macro_f1']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['test']['macro_f1']:.4f}")
    print(f"Test weighted F1: {metrics['test']['weighted_f1']:.4f}")
    print(f"Arrhythmia-only macro F1: {metrics['test']['arrhythmia_only_macro_f1']:.4f}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
