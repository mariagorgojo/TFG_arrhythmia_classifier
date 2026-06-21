"""Final five-class hybrid CNN trained with Optuna hyperparameter search.

This script is the most complete supervised model iteration in the project.
It uses the dataset where ``CurrentECG`` is kept as one of the five labels,
loads both ECG waveform samples and rhythm-derived numerical features, splits
the data into train/validation/test groups by device, lets Optuna suggest
hyperparameter combinations from a defined search space, trains the selected
model, and writes the final metrics and confusion matrices.

The important methodological idea is that the model is not selected using the
test set. The validation set is used to choose the best epoch and the best
hyperparameter configuration. The test set is kept aside and evaluated only at
the end, so it gives a more honest estimate of final performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / ".pythonlibs"

# The project was developed on Windows with some packages installed locally in
# `.pythonlibs`. This block makes the script able to find those packages when it
# is executed from the project folder, without requiring a global installation.
if LOCAL_SITE_PACKAGES.exists() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.append(str(LOCAL_SITE_PACKAGES))

# PyTorch, NumPy and SciPy may need native Windows DLL files. Adding these
# directories explicitly avoids import errors when the packages are stored in
# the local project dependency folder.
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hybrid_cnn1d_currentecg5_optuna_100trials"
MIN_RECOMMENDED_TRIALS = 100


@dataclass(frozen=True)
class HyperParameters:
    """One complete set of tunable choices for the model and training loop.

    A dataclass is used so that the values selected by Optuna can be passed
    around as one named object instead of as many independent variables. The
    class is frozen because these values should not change once a trial starts.
    """

    conv_layers: int
    base_channels: int
    kernel_size: int
    rhythm_hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int


class TunableHybridCNN(nn.Module):
    """Hybrid neural network that combines ECG waveform and rhythm features.

    The model has two branches:

    1. ECG branch: receives the 2,048 ECG samples and applies one-dimensional
       convolutional layers. This branch learns signal morphology and local
       temporal patterns.
    2. Rhythm branch: receives the manually extracted numerical features, such
       as RR interval statistics and marker fractions. This branch gives the
       model direct access to rhythm/timing information.

    The two learned representations are concatenated and sent to the final
    classifier, which outputs one score per class.
    """

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
        """Build all layers of the hybrid CNN.

        The layer sizes are controlled by Optuna through the arguments received
        here. This is why the class is called "Tunable": the same code can build
        a smaller or larger network depending on the selected hyperparameters.
        """

        super().__init__()
        padding = kernel_size // 2
        in_channels = 1
        conv_blocks: list[nn.Module] = []
        out_channels = base_channels

        # The ECG is a one-channel time series, so Conv1d is the natural
        # convolution type. Each block extracts temporal patterns, normalizes
        # activations, applies a non-linear function, and then downsamples the
        # signal length with max pooling.
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

        # Adaptive average pooling compresses the remaining time dimension to
        # one value per channel. This produces a fixed-size ECG representation
        # regardless of the exact temporal length after pooling.
        conv_blocks.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten()])
        self.ecg_branch = nn.Sequential(*conv_blocks)

        # The rhythm branch is a small fully connected network. LayerNorm helps
        # stabilize the numerical features after standardization, and dropout
        # reduces overfitting by randomly disabling part of the representation
        # during training.
        self.rhythm_branch = nn.Sequential(
            nn.Linear(num_rhythm_features, rhythm_hidden),
            nn.LayerNorm(rhythm_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(rhythm_hidden, rhythm_hidden),
            nn.ReLU(),
        )

        # The classifier receives both sources of information together. Its
        # output is a vector of logits: one unnormalized score for each class.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_channels + rhythm_hidden, num_classes),
        )

    def forward(self, ecg: torch.Tensor, rhythm_features: torch.Tensor) -> torch.Tensor:
        """Run one batch through the model and return class logits.

        PyTorch calls this method automatically when we write
        ``model(ecg, rhythm_features)``. The highest output logit corresponds to
        the predicted class before metrics are calculated.
        """

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
    """Create a PyTorch DataLoader for a selected subset of the dataset.

    ``indices`` selects whether we are building the train, validation or test
    subset. ``unsqueeze(1)`` adds the ECG channel dimension expected by Conv1d:
    the waveform changes from ``batch x samples`` to ``batch x 1 x samples``.
    """

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
    """Evaluate the model without updating its weights.

    This function is used for validation and test. ``model.eval()`` disables
    training-specific behavior such as dropout, and ``torch.no_grad()`` prevents
    PyTorch from storing gradients because we are only measuring performance.
    """

    model.eval()
    total_loss = 0.0
    predictions: list[int] = []
    true_labels: list[int] = []
    with torch.no_grad():
        for X_batch, rhythm_batch, y_batch in loader:
            # Move the current batch to CPU or GPU, depending on what is
            # available. In this project run, the recorded device was CPU.
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
    """Train the model for one full pass through the training subset.

    One epoch means that every training ECG has been seen once. During each
    batch, the model predicts logits, the loss function measures the error, the
    gradient is calculated, and the optimizer updates the model weights.
    """

    model.train()
    total_loss = 0.0
    predictions: list[int] = []
    true_labels: list[int] = []
    for X_batch, rhythm_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        rhythm_batch = rhythm_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        # Forward pass: compute predictions for the current batch.
        logits = model(X_batch, rhythm_batch)
        loss = loss_fn(logits, y_batch)

        # Backward pass: calculate how each trainable weight contributed to the
        # loss, then let AdamW update those weights.
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * y_batch.size(0)
        predictions.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        true_labels.extend(y_batch.detach().cpu().numpy().tolist())
    mean_loss = total_loss / len(loader.dataset)
    accuracy = float(accuracy_score(true_labels, predictions))
    return mean_loss, accuracy


def suggest_hyperparameters(
    trial: optuna.Trial,
    *,
    min_epochs: int,
    max_epochs: int,
) -> HyperParameters:
    """Define the hyperparameter search space explored by Optuna.

    The selected ranges are intentionally limited. The aim is not to search an
    enormous model space, but to compare realistic configurations that can be
    trained locally and defended methodologically in a final degree project.
    Every value is proposed by Optuna through `trial.suggest_*`; no fixed
    reference configurations are manually queued.
    """

    return HyperParameters(
        conv_layers=trial.suggest_int("conv_layers", 2, 4),
        base_channels=trial.suggest_categorical("base_channels", [8, 16, 32]),
        kernel_size=trial.suggest_categorical("kernel_size", [5, 7, 9, 11]),
        rhythm_hidden=trial.suggest_categorical("rhythm_hidden", [16, 32, 64]),
        dropout=trial.suggest_float("dropout", 0.10, 0.40, step=0.05),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4, 1e-3]),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        max_epochs=trial.suggest_int("max_epochs", min_epochs, max_epochs),
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
    """Train one candidate configuration and return its best validation score.

    This function is used twice:

    - during Optuna search, where each trial is a candidate model;
    - after Optuna, where the best candidate is trained as the final model.

    The quality criterion is validation macro F1, because the dataset is
    imbalanced and macro F1 gives each class the same importance.
    """

    set_reproducible_seed(seed)

    # Build independent loaders for training and validation. Training is
    # shuffled so the model does not see examples in the same order every epoch.
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

    # Create the model with the hyperparameters of this candidate trial.
    model = TunableHybridCNN(
        num_classes=len(labels),
        num_rhythm_features=rhythm_features.shape[1],
        conv_layers=params.conv_layers,
        base_channels=params.base_channels,
        kernel_size=params.kernel_size,
        rhythm_hidden=params.rhythm_hidden,
        dropout=params.dropout,
    ).to(device)

    # Class weights are calculated only from the training set. Minority classes
    # receive larger weights so their errors influence the loss more strongly.
    class_weights = compute_class_weights(y[train_index], len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW is used because it combines Adam optimization with decoupled weight
    # decay, which is a standard way to regularize neural networks.
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
        # Train one epoch, then evaluate on validation data that is not used to
        # update the weights.
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

        # During Optuna search, intermediate validation scores are reported so
        # the pruner can stop weak candidates early.
        if trial is not None:
            trial.report(macro_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Keep a copy of the best validation epoch. If later epochs get worse,
        # the model is restored to this best state before being returned.
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

    # Restore the best validation epoch instead of keeping the final epoch
    # automatically. This is a simple early-stopping strategy.
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
    """Write the numerical and visual evaluation outputs for one subset.

    The same function is used for validation and test outputs. It stores:
    a text classification report, a confusion matrix CSV, and a PNG image of
    the confusion matrix for easier inclusion in documentation.
    """

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
    """Export the Optuna trials so the hyperparameter search is auditable."""

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


def count_finished_trials(study: optuna.Study) -> int:
    """Count trials that already finished, whether completed, pruned or failed.

    This makes the optimization resumable. If the computer is stopped after 37
    finished trials, the next run can continue until the target total is reached
    instead of starting again from zero.
    """

    return sum(1 for trial in study.trials if trial.state.is_finished())


def write_progress_json(
    path: Path,
    study: optuna.Study,
    *,
    target_trials: int,
    storage_path: Path,
) -> None:
    """Save a small progress snapshot that can be inspected while training.

    This file is intentionally separate from the final `metrics.json`. It is
    updated during the Optuna search and is useful if training is interrupted
    before final evaluation has been generated.
    """

    finished_trials = count_finished_trials(study)
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best_trial = study.best_trial if complete_trials else None
    progress = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "target_trials": target_trials,
        "finished_trials": finished_trials,
        "remaining_trials": max(0, target_trials - finished_trials),
        "complete_trials": len(complete_trials),
        "pruned_trials": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
        ),
        "failed_trials": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL
        ),
        "running_trials": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING
        ),
        "best_trial_number": best_trial.number if best_trial is not None else None,
        "best_validation_macro_f1": best_trial.value if best_trial is not None else None,
        "best_hyperparameters": best_trial.params if best_trial is not None else None,
        "storage_path": str(storage_path),
    }
    path.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def build_optuna_storage_url(storage_path: Path) -> str:
    """Build a Windows-compatible SQLite URL for Optuna RDB storage."""

    return f"sqlite:///{storage_path.resolve().as_posix()}"


def run_optimization(
        dataset_path: Path,
        output_dir: Path,
        *,
        validation_size: float,
        test_size: float,
        random_state: int,
        n_trials: int,
        min_epochs: int,
        max_epochs: int,
        patience: int,
        split_strategy: str,
        study_name: str,
        storage_path: Path | None,
    ) -> dict[str, object]:
    """Run the complete final experiment from dataset loading to saved metrics.

    This is the main scientific workflow of the script:

    1. Load the prepared supervised dataset.
    2. Split it by device into train, validation and test.
    3. Standardize rhythm features using only the training set statistics.
    4. Use Optuna to suggest at least 100 hyperparameter combinations and select
       the best one with validation macro F1.
    5. Train the final model with the selected configuration.
    6. Evaluate validation and test subsets.
    7. Save the model checkpoint and all result files.
    """

    set_reproducible_seed(random_state)

    # Load arrays from the `.npz` dataset. X contains ECG samples, while
    # rhythm_features contains the additional numerical variables.
    dataset = load_training_dataset(dataset_path)
    X = dataset["X"]
    rhythm_features = dataset["rhythm_features"]
    rhythm_feature_names = dataset["rhythm_feature_names"]
    y = dataset["y"]
    labels = dataset["labels"]
    device_ids = dataset["device_ids"]

    # The split is performed by device, not by individual ECG record. This
    # avoids placing records from the same device in both training and test.
    train_index, validation_index, test_index = split_train_validation_test_by_device(
        X,
        y,
        device_ids,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        split_strategy=split_strategy,
    )

    # Standardization is fitted on the training set only, then applied to all
    # subsets. This prevents information leakage from validation/test data.
    rhythm_features, rhythm_mean, rhythm_std = standardize_rhythm_features(
        rhythm_features,
        train_index,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    if storage_path is None:
        storage_path = output_dir / "optuna_study.sqlite3"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    trials_csv_path = output_dir / "optuna_trials.csv"
    progress_json_path = output_dir / "optuna_progress.json"

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: train one candidate and return validation macro F1."""

        params = suggest_hyperparameters(
            trial,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
        )
        _, _, best_macro_f1 = train_candidate(
            X=X,
            rhythm_features=rhythm_features,
            y=y,
            labels=labels,
            train_index=train_index,
            validation_index=validation_index,
            params=params,
            epochs=params.max_epochs,
            patience=max(2, min(patience, params.max_epochs)),
            device=device,
            seed=random_state + trial.number,
            trial=trial,
        )
        return best_macro_f1

    # TPESampler proposes new trials based on previous trial performance. The
    # first startup trials are sampled without the TPE model so Optuna has an
    # initial set of observations. The median pruner can stop candidates that
    # are clearly underperforming after the warm-up epochs.
    startup_trials = min(10, max(1, n_trials))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=build_optuna_storage_url(storage_path),
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=random_state,
            n_startup_trials=startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )

    # No reference trials are enqueued here. Each candidate configuration is
    # generated inside `suggest_hyperparameters()` with Optuna's `trial.suggest`
    # API, which makes the search procedure independent from manually fixed
    # starting architectures.
    finished_before = count_finished_trials(study)
    remaining_trials = max(0, n_trials - finished_before)
    print(
        f"Optuna study: {study.study_name} | finished {finished_before}/{n_trials} "
        f"trials | remaining {remaining_trials}",
        flush=True,
    )
    print(f"Optuna SQLite storage: {storage_path}", flush=True)

    def progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Write progress files and print one visible line after each trial."""

        write_trials_csv(trials_csv_path, study)
        write_progress_json(
            progress_json_path,
            study,
            target_trials=n_trials,
            storage_path=storage_path,
        )
        finished_now = count_finished_trials(study)
        best_value = None
        complete_trials = [
            item
            for item in study.trials
            if item.state == optuna.trial.TrialState.COMPLETE
        ]
        if complete_trials:
            best_value = study.best_value
        print(
            "[Optuna progress] "
            f"{finished_now}/{n_trials} finished | "
            f"trial={trial.number} state={trial.state.name} "
            f"value={trial.value} best={best_value}",
            flush=True,
        )

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[progress_callback],
            gc_after_trial=True,
        )
    else:
        print(
            "Optuna search already reached the requested number of finished trials. "
            "Proceeding to final training/evaluation.",
            flush=True,
        )
    write_trials_csv(trials_csv_path, study)
    write_progress_json(
        progress_json_path,
        study,
        target_trials=n_trials,
        storage_path=storage_path,
    )

    # Train the final model using the best hyperparameters found by Optuna.
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise RuntimeError(
            "Optuna did not complete any trial. Rerun the script to continue the search."
        )
    best_params = HyperParameters(**study.best_trial.params)
    final_model, final_history, best_validation_macro_f1 = train_candidate(
        X=X,
        rhythm_features=rhythm_features,
        y=y,
        labels=labels,
        train_index=train_index,
        validation_index=validation_index,
        params=best_params,
        epochs=best_params.max_epochs,
        patience=patience,
        device=device,
        seed=random_state + study.best_trial.number,
    )

    # Recreate the weighted loss and deterministic loaders for final evaluation.
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

    # Store validation and test reports separately. Validation explains model
    # selection; test is the final held-out evaluation.
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

    # The checkpoint is useful for future inference. It includes not only the
    # learned weights, but also labels, feature names, preprocessing statistics
    # and the selected hyperparameters needed to reproduce predictions.
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

    # metrics.json is the compact experiment record: dataset, split sizes,
    # selected hyperparameters, class weights, training history and final
    # validation/test results.
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
        "minimum_recommended_trials": MIN_RECOMMENDED_TRIALS,
        "manual_reference_trials_used": False,
        "hyperparameter_suggestion_method": "All optimized values are generated with trial.suggest_*.",
        "optuna_sampler": "TPESampler",
        "optuna_startup_trials": startup_trials,
        "optuna_pruner": "MedianPruner",
        "optuna_storage_path": str(storage_path),
        "optuna_study_name": study.study_name,
        "finished_trials": count_finished_trials(study),
        "epoch_search_range": {
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
        },
        "selected_max_epochs": best_params.max_epochs,
        "final_epochs_requested": best_params.max_epochs,
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
            "trials_csv": str(trials_csv_path),
            "progress_json": str(progress_json_path),
            "optuna_storage": str(storage_path),
            "metrics_json": str(output_dir / "metrics.json"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for reproducible script execution."""

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
    parser.add_argument("--n-trials", type=int, default=MIN_RECOMMENDED_TRIALS)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--study-name",
        default="currentecg5_hybrid_cnn_100trials",
        help="Optuna study name. Reuse the same name to resume the same search.",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=None,
        help=(
            "SQLite file used by Optuna to save trials progressively. "
            "Defaults to <output-dir>/optuna_study.sqlite3."
        ),
    )
    return parser


def main() -> int:
    """Parse command-line arguments, run the experiment, and print a summary."""

    parser = build_parser()
    args = parser.parse_args()
    if args.min_epochs > args.max_epochs:
        parser.error("--min-epochs must be lower than or equal to --max-epochs")
    if args.n_trials < MIN_RECOMMENDED_TRIALS:
        print(
            "Warning: fewer than 100 trials were requested. This is useful for a "
            "quick technical check, but the defended experiment should use at "
            "least 100 trials."
        )
    metrics = run_optimization(
        args.dataset_path,
        args.output_dir,
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
        n_trials=args.n_trials,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        split_strategy=args.split_strategy,
        study_name=args.study_name,
        storage_path=args.storage_path,
    )
    print(f"Model: {metrics['model_path']}")
    print(f"Best hyperparameters: {metrics['best_hyperparameters']}")
    print(f"Selected max epochs: {metrics['selected_max_epochs']}")
    print(f"Validation macro F1: {metrics['validation']['macro_f1']:.4f}")
    print(f"Test accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['test']['macro_f1']:.4f}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
