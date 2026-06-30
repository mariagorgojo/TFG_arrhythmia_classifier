"""Five-class hybrid CNN + bidirectional-LSTM trained with Optuna (comparison model).

This script is an alternative architecture to ``hybrid_cnn1d_currentecg5_optuna.py``,
kept for a like-for-like comparison after the supervisor suggested testing a recurrent
model. The waveform branch first applies a few one-dimensional convolutional blocks to
shorten the signal, and then a bidirectional LSTM that reads the shortened sequence in
both directions. The rhythm-feature branch is unchanged.

Everything else follows exactly the same protocol as the CNN model — the device-disjoint
train/validation/test split, the class weighting, the model selection by validation macro
F1, and the single held-out test evaluation — so the two architectures are directly
comparable.

The search budget can be increased with ``--n-trials``. For a long run on Google Colab,
pass ``--storage auto`` so the Optuna study is saved to disk: if the session disconnects,
re-running the same command reopens the study and continues toward the same total number
of trials, instead of starting over.

Run (Colab/local):
    python -m arrhythmia_classifier.hybrid_cnn_bilstm_optuna \
        --dataset-path data/processed/training_dataset/ecg_training_dataset_currentecg5_features.npz \
        --output-dir results/hybrid_cnn_bilstm_currentecg5 \
        --n-trials 150 --min-epochs 6 --max-epochs 20 --patience 5 --storage auto

The epoch range (6-20) and patience (5) match the CNN script, so the two architectures are
searched under the same training schedule.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Project code first: importing these prepares local dependency paths on Windows
# (the shim lives in cnn1d_classifier) before the heavy libraries load.
from arrhythmia_classifier.cnn1d_classifier import (
    compute_class_weights,
    set_reproducible_seed,
)
from arrhythmia_classifier.hybrid_cnn1d_classifier import (
    load_training_dataset,
    standardize_rhythm_features,
)
from arrhythmia_classifier.hybrid_cnn1d_train_val_test import (
    split_train_validation_test_by_device,
)

# The training loop, the data loader and the evaluation/report helpers do not depend on
# the model architecture, so they are imported from the CNN script and reused unchanged.
# This keeps the comparison fair (identical training and evaluation code) and keeps this
# file short.
from arrhythmia_classifier.hybrid_cnn1d_currentecg5_optuna import (
    evaluate,
    make_loader,
    train_one_epoch,
    write_report_outputs,
)

import numpy as np
import optuna
import torch
from torch import nn
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Test macro F1 of the final CNN model (the 150-trial run), printed next to the
# CNN-BiLSTM result so the two architectures can be compared at a glance.
CNN_REFERENCE_TEST_MACRO_F1 = 0.966


@dataclass(frozen=True)
class BiLSTMHyperParameters:
    """One complete set of tunable choices for the CNN-BiLSTM model and its training.

    Like the CNN script, a frozen dataclass is used so that the values chosen by Optuna
    travel together as one named object instead of many loose variables, and so that they
    cannot be changed once a trial has started.
    """

    conv_layers: int
    base_channels: int
    kernel_size: int
    lstm_hidden: int
    lstm_layers: int
    rhythm_hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int


class HybridCNNBiLSTM(nn.Module):
    """Hybrid network that reads the ECG waveform with a CNN + bidirectional LSTM.

    The model has two branches, exactly like the CNN model:

    1. Waveform branch: a few one-dimensional convolutional blocks shorten the ECG and
       extract local patterns, and then a bidirectional LSTM reads the shortened
       sequence. An LSTM is a recurrent network: it processes the sequence one step at a
       time and carries an internal memory (the hidden state) from step to step, which is
       a natural way to model the temporal structure of a heartbeat. "Bidirectional"
       means two LSTMs run over the sequence, one from start to end and one from end to
       start, so every position has context from both sides.
    2. Rhythm branch: a small fully connected network over the manually extracted
       numerical features (RR-interval statistics, marker fractions, etc.).

    The two learned representations are concatenated and sent to the final classifier,
    which outputs one score per class. The ``forward(ecg, rhythm_features)`` signature is
    the same as the CNN model, which is why the shared data loaders and training loop can
    be reused without changes.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        num_rhythm_features: int,
        conv_layers: int,
        base_channels: int,
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        rhythm_hidden: int,
        dropout: float,
    ) -> None:
        """Build all layers of the model from the hyperparameters chosen by Optuna."""

        super().__init__()
        padding = kernel_size // 2
        in_channels = 1
        conv_blocks: list[nn.Module] = []
        out_channels = base_channels

        # Convolutional front-end. Each block extracts temporal patterns, normalizes the
        # activations, applies a non-linear function, and then halves the signal length
        # with max pooling. After the blocks the LSTM receives a shorter, richer sequence,
        # which is faster and more stable than running an LSTM over the 2,048 raw samples.
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
        self.conv = nn.Sequential(*conv_blocks)

        # Bidirectional LSTM over the convolved sequence. The dropout argument of nn.LSTM
        # only acts between stacked layers, so it is disabled when there is a single layer
        # (PyTorch warns otherwise).
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        # The waveform representation is the forward and backward final hidden states
        # joined together, so its size is twice the LSTM hidden size.
        waveform_dim = 2 * lstm_hidden

        # Rhythm-feature branch (identical to the CNN model). LayerNorm stabilizes the
        # standardized numerical features and dropout reduces overfitting.
        self.rhythm_branch = nn.Sequential(
            nn.Linear(num_rhythm_features, rhythm_hidden),
            nn.LayerNorm(rhythm_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(rhythm_hidden, rhythm_hidden),
            nn.ReLU(),
        )

        # The classifier receives both sources together and returns one logit per class.
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(waveform_dim + rhythm_hidden, num_classes),
        )

    def forward(self, ecg: torch.Tensor, rhythm_features: torch.Tensor) -> torch.Tensor:
        """Run one batch through the model and return class logits."""

        # The convolutional front-end returns a tensor shaped (batch, channels, length).
        conv_output = self.conv(ecg)

        # nn.LSTM expects (batch, time-steps, features), so the length axis is moved to
        # the middle: each time-step becomes a feature vector of size "channels".
        sequence = conv_output.permute(0, 2, 1)

        # The LSTM returns the output at every step and the final hidden state h_n. For a
        # bidirectional LSTM, h_n[-2] is the last layer read forwards and h_n[-1] is the
        # same layer read backwards. Concatenating them summarizes the whole waveform from
        # both directions.
        _, (h_n, _) = self.lstm(sequence)
        forward_last = h_n[-2]
        backward_last = h_n[-1]
        waveform_embedding = torch.cat([forward_last, backward_last], dim=1)

        rhythm_embedding = self.rhythm_branch(rhythm_features)
        return self.classifier(torch.cat([waveform_embedding, rhythm_embedding], dim=1))


def suggest_hyperparameters(
    trial: optuna.Trial,
    *,
    min_epochs: int,
    max_epochs: int,
) -> BiLSTMHyperParameters:
    """Define the hyperparameter search space explored by Optuna for the CNN-BiLSTM.

    The ranges are intentionally compact. The goal is not to search an enormous space but
    to compare realistic configurations against the CNN model under the same protocol.
    Every value is proposed by Optuna through ``trial.suggest_*``.
    """

    return BiLSTMHyperParameters(
        conv_layers=trial.suggest_int("conv_layers", 2, 3),
        base_channels=trial.suggest_categorical("base_channels", [8, 16]),
        kernel_size=trial.suggest_categorical("kernel_size", [5, 7]),
        lstm_hidden=trial.suggest_categorical("lstm_hidden", [32, 64]),
        lstm_layers=trial.suggest_int("lstm_layers", 1, 2),
        rhythm_hidden=trial.suggest_categorical("rhythm_hidden", [32, 64]),
        dropout=trial.suggest_float("dropout", 0.10, 0.40, step=0.10),
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
    params: BiLSTMHyperParameters,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
    trial: optuna.Trial | None = None,
) -> tuple[nn.Module, float]:
    """Train one CNN-BiLSTM candidate and return its best-validation model and macro F1.

    This function is used twice, exactly like in the CNN script: during the Optuna search,
    where each trial is a candidate model, and afterwards, where the best candidate is
    retrained as the final model. The quality criterion is validation macro F1, because
    the dataset is imbalanced and macro F1 gives every class the same importance.
    """

    set_reproducible_seed(seed)

    # Build independent loaders. Training is shuffled so the model does not see the
    # examples in the same order every epoch; validation is not shuffled.
    train_loader = make_loader(
        X, rhythm_features, y, train_index, batch_size=params.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        X, rhythm_features, y, validation_index, batch_size=params.batch_size, shuffle=False
    )

    # Create the model with the hyperparameters of this candidate.
    model = HybridCNNBiLSTM(
        num_classes=len(labels),
        num_rhythm_features=rhythm_features.shape[1],
        conv_layers=params.conv_layers,
        base_channels=params.base_channels,
        kernel_size=params.kernel_size,
        lstm_hidden=params.lstm_hidden,
        lstm_layers=params.lstm_layers,
        rhythm_hidden=params.rhythm_hidden,
        dropout=params.dropout,
    ).to(device)

    # Class weights are computed only from the training set; minority classes get larger
    # weights so their errors count more in the loss. AdamW is Adam with decoupled weight
    # decay, a standard way to regularize neural networks.
    class_weights = compute_class_weights(y[train_index], len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    best_macro_f1 = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # Train one epoch, then measure macro F1 on the validation set.
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        _, _, y_true, y_pred = evaluate(model, validation_loader, loss_fn, device)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # During the Optuna search, report the intermediate score so the pruner can stop
        # clearly weak candidates early.
        if trial is not None:
            trial.report(macro_f1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Keep a copy of the best validation epoch. If later epochs get worse for
        # "patience" epochs in a row, stop early and restore this best state.
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
    return model, best_macro_f1


def run(args: argparse.Namespace) -> None:
    """Run the whole comparison experiment: search, retrain the best, evaluate on test."""

    set_reproducible_seed(args.random_state)

    # Load the prepared dataset. X holds the ECG samples and rhythm_features the extra
    # numerical variables.
    dataset = load_training_dataset(args.dataset_path)
    X = dataset["X"]
    rhythm_features = dataset["rhythm_features"]
    y = dataset["y"]
    labels = dataset["labels"]
    device_ids = dataset["device_ids"]

    # Split by device so no device appears in more than one subset, then standardize the
    # rhythm features using training statistics only (to avoid information leakage).
    train_index, validation_index, test_index = split_train_validation_test_by_device(
        X, y, device_ids,
        validation_size=args.validation_size, test_size=args.test_size,
        random_state=args.random_state, split_strategy=args.split_strategy,
    )
    rhythm_features, rhythm_mean, rhythm_std = standardize_rhythm_features(
        rhythm_features, train_index
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Device: {device} | train/val/test = "
        f"{train_index.size}/{validation_index.size}/{test_index.size}"
    )

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: train one candidate and return its validation macro F1."""

        params = suggest_hyperparameters(
            trial, min_epochs=args.min_epochs, max_epochs=args.max_epochs
        )
        _, best_macro_f1 = train_candidate(
            X=X, rhythm_features=rhythm_features, y=y, labels=labels,
            train_index=train_index, validation_index=validation_index,
            params=params, epochs=params.max_epochs,
            patience=max(2, min(args.patience, params.max_epochs)),
            device=device, seed=args.random_state + trial.number, trial=trial,
        )
        return best_macro_f1

    # TPESampler proposes new trials from the previous ones; the median pruner stops
    # clearly underperforming candidates after a few warm-up epochs.
    sampler = optuna.samplers.TPESampler(
        seed=args.random_state, n_startup_trials=min(8, args.n_trials)
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)

    # When a storage path is given, the study is saved on disk instead of only in memory.
    # This makes a long Colab run resumable: re-running the same command reopens the same
    # study and continues toward the same total --n-trials. Without --storage the study is
    # kept in memory, as before.
    if args.storage:
        if args.storage == "auto":
            # Put the database file inside the output directory (point --output-dir at
            # Drive on Colab so the file survives a disconnect).
            storage_path = args.output_dir / "bilstm_study.sqlite3"
            storage_url = f"sqlite:///{storage_path.resolve()}"
        else:
            storage_url = args.storage
        study = optuna.create_study(
            study_name=args.study_name, direction="maximize",
            sampler=sampler, pruner=pruner,
            storage=storage_url, load_if_exists=True,
        )
        # Count the trials that are already finished so only the remaining ones are run.
        already_done = 0
        for trial in study.trials:
            if trial.state.is_finished():
                already_done += 1
        remaining_trials = max(0, args.n_trials - already_done)
        print(f"Storage: {storage_url}")
        if already_done > 0:
            print(
                f"Resuming study '{args.study_name}': {already_done} trials already done, "
                f"{remaining_trials} remaining (target {args.n_trials})."
            )
    else:
        study = optuna.create_study(
            study_name=args.study_name, direction="maximize",
            sampler=sampler, pruner=pruner,
        )
        remaining_trials = args.n_trials

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, gc_after_trial=True)
    else:
        print(f"Target of {args.n_trials} trials already reached; retraining the best configuration.")

    # Retrain the best configuration found by Optuna as the final model.
    best_params = BiLSTMHyperParameters(**study.best_trial.params)
    final_model, val_macro_f1 = train_candidate(
        X=X, rhythm_features=rhythm_features, y=y, labels=labels,
        train_index=train_index, validation_index=validation_index,
        params=best_params, epochs=best_params.max_epochs, patience=args.patience,
        device=device, seed=args.random_state + study.best_trial.number,
    )

    # Single held-out test evaluation.
    class_weights = compute_class_weights(y[train_index], len(labels)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    test_loader = make_loader(
        X, rhythm_features, y, test_index, batch_size=best_params.batch_size, shuffle=False
    )
    _, test_acc, y_true, y_pred = evaluate(final_model, test_loader, loss_fn, device)
    test_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    test_weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Arrhythmia-only macro F1: average the F1 of the four arrhythmia classes, excluding
    # the easy and dominant CurrentECG baseline class.
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=list(range(len(labels))), zero_division=0
    )
    arrhythmia_indices = []
    for index, label in enumerate(labels):
        if str(label) != "CurrentECG":
            arrhythmia_indices.append(index)
    arrhythmia_macro_f1 = float(np.mean([per_class_f1[i] for i in arrhythmia_indices]))

    # Write the confusion matrix and classification report (same helper as the CNN model).
    test_report, _ = write_report_outputs(
        output_dir=args.output_dir, prefix="test",
        title="CNN-BiLSTM CurrentECG5 Final Test Confusion Matrix",
        y_true=y_true, y_pred=y_pred, labels=labels,
    )

    # Save the trained model together with everything needed to reproduce predictions.
    model_path = args.output_dir / "hybrid_cnn_bilstm_currentecg5_model.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "architecture": "HybridCNNBiLSTM",
            "labels": labels.tolist(),
            "hyperparameters": asdict(best_params),
            "rhythm_feature_mean": rhythm_mean.tolist(),
            "rhythm_feature_std": rhythm_std.tolist(),
            "target_samples": int(X.shape[1]),
        },
        model_path,
    )

    # Compact record of the run, including the CNN reference for the comparison.
    metrics = {
        "model_type": "HybridCNNBiLSTM",
        "n_trials": args.n_trials,
        "best_hyperparameters": asdict(best_params),
        "validation_macro_f1": val_macro_f1,
        "test": {
            "accuracy": test_acc,
            "macro_f1": test_macro_f1,
            "weighted_f1": test_weighted_f1,
            "arrhythmia_only_macro_f1": arrhythmia_macro_f1,
            "classification_report": test_report,
        },
        "cnn_reference_test_macro_f1": CNN_REFERENCE_TEST_MACRO_F1,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Print a short summary, with the CNN result next to the CNN-BiLSTM one.
    print("\n=== CNN-BiLSTM (final test) ===")
    print(f"best hyperparameters: {asdict(best_params)}")
    print(f"validation macro F1 : {val_macro_f1:.4f}")
    print(f"test accuracy       : {test_acc:.4f}")
    print(f"test macro F1       : {test_macro_f1:.4f}   (CNN reference: {CNN_REFERENCE_TEST_MACRO_F1})")
    print(f"test weighted F1    : {test_weighted_f1:.4f}")
    print(f"arrhythmia-only F1  : {arrhythmia_macro_f1:.4f}")
    print(f"saved: {model_path}")
    print(f"saved: {args.output_dir / 'metrics.json'}")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for reproducible script execution."""

    parser = argparse.ArgumentParser(description="Train a CNN-BiLSTM hybrid as a comparison model.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "hybrid_cnn_bilstm_currentecg5")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("stratified_group", "group_shuffle"), default="stratified_group")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Total target number of trials (with --storage, trials already done are counted).",
    )
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--storage", type=str, default=None,
        help=(
            "Optuna storage to make the search resumable. Pass 'auto' to put a SQLite file "
            "in --output-dir, or an explicit URL "
            "(e.g. sqlite:////content/drive/MyDrive/.../bilstm_study.sqlite3). "
            "Omit for an in-memory study."
        ),
    )
    parser.add_argument("--study-name", type=str, default="currentecg5_cnn_bilstm")
    return parser


def main() -> int:
    """Parse command-line arguments and run the experiment."""

    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
