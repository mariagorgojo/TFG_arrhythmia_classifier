from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
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
import optuna
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
from arrhythmia_classifier.hybrid_cnn1d_classifier import (
    load_training_dataset,
    standardize_rhythm_features,
)
from arrhythmia_classifier.hybrid_cnn1d_train_val_test import (
    split_train_validation_test_by_device,
)


DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "training_dataset"
    / "ecg_training_dataset_currentecg5_features.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid_cnn1d_currentecg5_optuna"

REFERENCE_TRIALS = [
    {
        "conv_layers": 3,
        "base_channels": 8,
        "kernel_size": 11,
        "rhythm_hidden": 32,
        "dropout": 0.40,
        "learning_rate": 0.0017,
        "weight_decay": 0.001,
        "batch_size": 64,
    },
    {
        "conv_layers": 2,
        "base_channels": 32,
        "kernel_size": 7,
        "rhythm_hidden": 64,
        "dropout": 0.35,
        "learning_rate": 0.0024,
        "weight_decay": 0.0001,
        "batch_size": 256,
    },
    {
        "conv_layers": 4,
        "base_channels": 8,
        "kernel_size": 9,
        "rhythm_hidden": 32,
        "dropout": 0.40,
        "learning_rate": 0.0028,
        "weight_decay": 0.0001,
        "batch_size": 64,
    },
]


@dataclass(frozen=True)
class HyperParameters:
    conv_layers: int
    base_channels: int
    kernel_size: int
    rhythm_hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int


class TunableHybridCNN(nn.Module):
    """Hybrid CNN with Optuna-controlled capacity."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_rhythm_features: int,
        conv_layers: int,
        base_channels: int,
        kernel_size: int,
        rhythm_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        in_channels = 1
        conv_blocks: list[nn.Module] = []
        out_channels = base_channels
        for layer_index in range(conv_layers):
            out_channels = min(base_channels * (2**layer_index), 128)
            conv_blocks.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        conv_blocks.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
        self.ecg_branch = nn.Sequential(*conv_blocks)
        self.rhythm_branch = nn.Sequential(
            nn.Linear(num_rhythm_features, rhythm_hidden),
            nn.LayerNorm(rhythm_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(rhythm_hidden, rhythm_hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_channels + rhythm_hidden, num_classes),
        )

    def forward(self, ecg: torch.Tensor, rhythm_features: torch.Tensor) -> torch.Tensor:
        ecg_embedding = self.ecg_branch(ecg)
        rhythm_embedding = self.rhythm_branch(rhythm_features)
        return self.classifier(torch.cat([ecg_embedding, rhythm_embedding], dim=1))


def make_loader(
    X: np.ndarray,
    rhythm_features: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X[indices], dtype=torch.float32).unsqueeze(1),
        torch.tensor(rhythm_features[indices], dtype=torch.float32),
        torch.tensor(y[indices], dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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


def suggest_hyperparameters(trial: optuna.Trial) -> HyperParameters:
    return HyperParameters(
        conv_layers=trial.suggest_int("conv_layers", 2, 4),
        base_channels=trial.suggest_categorical("base_channels", [8, 16, 32]),
        kernel_size=trial.suggest_categorical("kernel_size", [5, 7, 9, 11]),
        rhythm_hidden=trial.suggest_categorical("rhythm_hidden", [16, 32, 64]),
        dropout=trial.suggest_float("dropout", 0.10, 0.40, step=0.05),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
    )


def train_candidate(
    *,
    X: np.ndarray,
    rhythm_features: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    params: HyperParameters,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[nn.Module, list[dict[str, float | int]], float]:
    set_reproducible_seed(seed)
    train_loader = make_loader(
        X, rhythm_features, y, train_index, batch_size=params.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        X,
        rhythm_features,
        y,
        validation_index,
        batch_size=params.batch_size,
        shuffle=False,
    )
    model = TunableHybridCNN(
        num_classes=len(labels),
        num_rhythm_features=rhythm_features.shape[1],
        conv_layers=params.conv_layers,
        base_channels=params.base_channels,
        kernel_size=params.kernel_size,
        rhythm_hidden=params.rhythm_hidden,
        dropout=params.dropout,
    ).to(device)
    class_weights = compute_class_weights(y[train_index], len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        validation_loss, validation_accuracy, y_true, y_pred = evaluate(
            model, validation_loader, loss_fn, device
        )
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": macro_f1,
                "validation_weighted_f1": weighted_f1,
            }
        )
        if trial is not None:
            trial.report(macro_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_macro_f1


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


def write_trials_csv(path: Path, study: optuna.Study) -> None:
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                **trial.params,
            }
        )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_optimization(
    dataset_path: Path,
    output_dir: Path,
    *,
    validation_size: float,
    test_size: float,
    random_state: int,
    n_trials: int,
    trial_epochs: int,
    final_epochs: int,
    patience: int,
    split_strategy: str,
) -> dict[str, object]:
    set_reproducible_seed(random_state)
    dataset = load_training_dataset(dataset_path)
    X = dataset["X"]
    rhythm_features = dataset["rhythm_features"]
    rhythm_feature_names = dataset["rhythm_feature_names"]
    y = dataset["y"]
    labels = dataset["labels"]
    device_ids = dataset["device_ids"]
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)
        _, _, best_macro_f1 = train_candidate(
            X=X,
            rhythm_features=rhythm_features,
            y=y,
            labels=labels,
            train_index=train_index,
            validation_index=validation_index,
            params=params,
            epochs=trial_epochs,
            patience=max(2, min(patience, trial_epochs)),
            device=device,
            seed=random_state + trial.number,
            trial=trial,
        )
        return best_macro_f1

    startup_trials = min(2, max(1, n_trials))
    study = optuna.create_study(
        study_name="currentecg5_hybrid_cnn",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state,
            n_startup_trials=startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    for reference_trial in REFERENCE_TRIALS[:n_trials]:
        study.enqueue_trial(reference_trial)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    write_trials_csv(output_dir / "optuna_trials.csv", study)

    best_params = HyperParameters(**study.best_trial.params)
    final_model, final_history, best_validation_macro_f1 = train_candidate(
        X=X,
        rhythm_features=rhythm_features,
        y=y,
        labels=labels,
        train_index=train_index,
        validation_index=validation_index,
        params=best_params,
        epochs=final_epochs,
        patience=patience,
        device=device,
        seed=random_state + study.best_trial.number,
    )
    class_weights = compute_class_weights(y[train_index], len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    validation_loader = make_loader(
        X,
        rhythm_features,
        y,
        validation_index,
        batch_size=best_params.batch_size,
        shuffle=False,
    )
    test_loader = make_loader(
        X,
        rhythm_features,
        y,
        test_index,
        batch_size=best_params.batch_size,
        shuffle=False,
    )
    validation_loss, validation_accuracy, validation_true, validation_pred = evaluate(
        final_model, validation_loader, loss_fn, device
    )
    test_loss, test_accuracy, test_true, test_pred = evaluate(
        final_model, test_loader, loss_fn, device
    )
    validation_report, validation_outputs = write_report_outputs(
        output_dir=output_dir,
        prefix="validation",
        title="CurrentECG5 Hybrid CNN Validation Confusion Matrix",
        y_true=validation_true,
        y_pred=validation_pred,
        labels=labels,
    )
    test_report, test_outputs = write_report_outputs(
        output_dir=output_dir,
        prefix="test",
        title="CurrentECG5 Hybrid CNN Final Test Confusion Matrix",
        y_true=test_true,
        y_pred=test_pred,
        labels=labels,
    )
    model_path = output_dir / "hybrid_cnn1d_currentecg5_optuna_model.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "labels": labels.tolist(),
            "dataset_path": str(dataset_path),
            "target_samples": int(X.shape[1]),
            "rhythm_feature_names": rhythm_feature_names.tolist(),
            "rhythm_feature_mean": rhythm_mean.tolist(),
            "rhythm_feature_std": rhythm_std.tolist(),
            "random_state": random_state,
            "architecture": "TunableHybridCNN",
            "hyperparameters": asdict(best_params),
            "evaluation_protocol": "device_disjoint_train_validation_test_with_optuna",
            "selection_metric": "validation_macro_f1",
        },
        model_path,
    )
    label_counts = lambda idx: dict(Counter(labels[y[idx]]))
    class_weight_values = class_weights.detach().cpu().numpy()
    metrics = {
        "model_type": "TunableHybridCNN",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "evaluation_protocol": "device_disjoint_train_validation_test_with_optuna",
        "currentecg_assumption": (
            "CurrentECG is used as a supervised label and treated operationally "
            "as the device/export baseline-rhythm class, while preserving the "
            "original label name."
        ),
        "limitation": (
            "Some CurrentECG strips may contain non-NSR rhythms or artifact. "
            "This introduces label noise that should be reviewed clinically."
        ),
        "validation_size_requested": validation_size,
        "test_size_requested": test_size,
        "split_strategy": split_strategy,
        "random_state": random_state,
        "n_trials": n_trials,
        "optuna_sampler": "TPESampler",
        "optuna_startup_trials": startup_trials,
        "optuna_pruner": "MedianPruner",
        "reference_trials": REFERENCE_TRIALS[:n_trials],
        "trial_epochs": trial_epochs,
        "final_epochs_requested": final_epochs,
        "final_epochs_completed": len(final_history),
        "best_trial_number": study.best_trial.number,
        "best_trial_seed": random_state + study.best_trial.number,
        "best_trial_validation_macro_f1": study.best_value,
        "best_final_validation_macro_f1": best_validation_macro_f1,
        "best_hyperparameters": asdict(best_params),
        "target_samples": int(X.shape[1]),
        "rhythm_feature_names": rhythm_feature_names.tolist(),
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
        "final_history": final_history,
        "validation": {
            "loss": validation_loss,
            "accuracy": validation_accuracy,
            "macro_f1": float(
                f1_score(validation_true, validation_pred, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(validation_true, validation_pred, average="weighted", zero_division=0)
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
            "classification_report": test_report,
            "outputs": test_outputs,
        },
        "device": str(device),
        "outputs": {
            "trials_csv": str(output_dir / "optuna_trials.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize and train a five-class hybrid CNN using CurrentECG as a supervised label."
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
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--trial-epochs", type=int, default=8)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = run_optimization(
        args.dataset_path,
        args.output_dir,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        n_trials=args.n_trials,
        trial_epochs=args.trial_epochs,
        final_epochs=args.final_epochs,
        patience=args.patience,
        split_strategy=args.split_strategy,
    )
    print(f"Model: {metrics['model_path']}")
    print(f"Best hyperparameters: {metrics['best_hyperparameters']}")
    print(f"Validation macro F1: {metrics['validation']['macro_f1']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['test']['macro_f1']:.4f}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
