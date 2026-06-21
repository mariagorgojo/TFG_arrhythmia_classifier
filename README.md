# TFG Arrhythmia Classifier

Prototype for classifying arrhythmia-related ECG occurrence records from
single-lead Medtronic XML exports.

The current active iteration uses five supervised labels:

```text
CurrentECG
AF
Asystole
Brady
VT
```

`CurrentECG` is kept as the original device/export label. In the current
iteration it is treated operationally as the baseline/non-arrhythmic class under
the tutor's assumption, but it is not renamed to clinically confirmed NSR.

The current model is a hybrid 1D CNN optimized with Optuna. It combines:

- a 2,048-sample ECG waveform window;
- rhythm and marker features extracted from the same occurrence record.

The previous four-arrhythmia model is preserved as historical comparison.

## Install

Create and activate a Python environment:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To run the reproducible extractor test:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Dataset

The historical four-arrhythmia shareable training-dataset candidate is:

```text
datasets/ecg_training_dataset_arrhythmia4_features_deidentified.npz
```

It contains ECG signals, rhythm features, labels and generic device groups for
device-separated evaluation. Original occurrence identifiers were removed.

Important: ECG signals remain clinical data even after identifier removal.
Obtain the appropriate data-governance approval before uploading or sharing the
`.npz` file. The repository ignores this file by default. If the reviewed file
is not committed, an authorized collaborator must place it in `datasets/`
before training.

The current five-class `CurrentECG` dataset is generated locally from the
private XML-derived occurrence data and is not included in GitHub:

```text
data/processed/training_dataset/ecg_training_dataset_currentecg5_features.npz
```

## Train The Hybrid Model

For the current five-class protocol with `CurrentECG` as a supervised label, the
hyperparameter search and model selection are run with:

```powershell
python -m arrhythmia_classifier.hybrid_cnn1d_currentecg5_optuna --dataset-path data\processed\training_dataset\ecg_training_dataset_currentecg5_features.npz --output-dir results\hybrid_cnn1d_currentecg5_optuna_100trials --n-trials 100 --min-epochs 6 --max-epochs 20 --patience 5
```

This protocol:

- uses train/validation/test groups separated by device;
- lets Optuna suggest hyperparameters and selects the best configuration by validation macro F1;
- retrains the selected configuration and evaluates the test groups once at the end;
- stores aggregate reports and confusion matrices under `results/`.

The final model reported in the thesis extends this search to 150 trials (an
additional 50 trials with a longer median-pruner warm-up) and retrains the best
configuration. On the held-out test set it reaches accuracy 0.991 and macro F1
0.966 (arrhythmia-only macro F1 0.958). The resulting checkpoint, metrics and
confusion matrix are stored under
`results\hybrid_cnn1d_currentecg5_optuna_150trials\`; the two-stage procedure is
described in `ADDED_SCRIPTS.md`.

The required private dataset is not included in GitHub.

For the final device-disjoint train/validation/test protocol:

```powershell
python -m arrhythmia_classifier.hybrid_cnn1d_train_val_test --dataset-path datasets\ecg_training_dataset_arrhythmia4_features_deidentified.npz --output-dir results\hybrid_cnn1d_arrhythmia4_train_val_test
```

This final protocol:

- learns model parameters from train device groups;
- selects the best checkpoint using validation macro F1;
- evaluates the test groups once at the end of the corrected run.

The test groups match the historical two-way prototype holdout. The reported
score is an internal evaluation, not an external validation cohort.

The earlier two-way prototype is preserved for historical comparison:

```powershell
python -m arrhythmia_classifier.hybrid_cnn1d_classifier --dataset-path datasets\ecg_training_dataset_arrhythmia4_features_deidentified.npz --output-dir results\hybrid_cnn1d_arrhythmia4
```

The script:

- separates train/test records by generic device group;
- uses class weights to reduce majority-class bias;
- selects the best checkpoint using macro F1;
- writes a classification report and confusion matrix.

## Train The Waveform-Only Baseline

```powershell
python -m arrhythmia_classifier.cnn1d_classifier --dataset-path datasets\ecg_training_dataset_arrhythmia4_features_deidentified.npz --output-dir results\cnn1d_arrhythmia4_baseline
```

## Figures

Utility scripts generate the per-class and error-analysis figures used in the report:

```powershell
python -m arrhythmia_classifier.plot_class_examples --dataset-path data\processed\training_dataset\ecg_training_dataset_currentecg5_features.npz --output-path figures\class_examples.png
python -m arrhythmia_classifier.plot_misclassified --dataset-path data\processed\training_dataset\ecg_training_dataset_currentecg5_features.npz --model-path results\hybrid_cnn1d_currentecg5_optuna_150trials\hybrid_cnn1d_currentecg5_optuna_model.pt --output-path figures\misclassified_asystole_brady.png
```

`plot_class_examples` plots one real ECG strip per class from the prepared dataset.
`plot_misclassified` plots test-set records misclassified between two classes
(default Asystole versus Brady) using the trained checkpoint. See `ADDED_SCRIPTS.md`.

## Historical Exploratory CurrentECG Inference

`CurrentECG` inference requires the private occurrence manifest and extracted
occurrence arrays. These files are not included in GitHub. The trained
checkpoint is also ignored by default and must be placed in `model/` after the
appropriate review.

Example:

```powershell
python -m arrhythmia_classifier.currentecg_inference --manifest-path D:\private_data\occurrence_dataset\occurrences_manifest.csv --model-path model\hybrid_cnn1d_train_val_test_model.pt --output-dir results\currentecg_exploratory
```

This section belongs to the previous four-arrhythmia protocol. The output is
exploratory only because that older classifier has no confirmed NSR/normal or
other/indeterminate class and is forced to select one of the four available
arrhythmia labels.

## Rebuild From Private XML Files

Raw XML/PDD files are intentionally excluded from GitHub.

To rebuild occurrence arrays from an authorized local XML folder:

```powershell
python -m arrhythmia_classifier.dataset_builder --xml-dir D:\private_data\pdds --output-dir D:\private_data\occurrence_dataset
```

To build the four-class feature dataset:

```powershell
python -m arrhythmia_classifier.dataset_loader --manifest-path D:\private_data\occurrence_dataset\occurrences_manifest.csv --label-strategy arrhythmia4 --output-path D:\private_data\ecg_training_dataset_arrhythmia4_features.npz --max-per-label 0
```

To build the current five-class feature dataset:

```powershell
python -m arrhythmia_classifier.dataset_loader --manifest-path D:\private_data\occurrence_dataset\occurrences_manifest.csv --label-strategy clinical5 --output-path data\processed\training_dataset\ecg_training_dataset_currentecg5_features.npz --max-per-label 0 --selection-strategy device_round_robin
```

## Repository Contents

```text
arrhythmia_classifier/   active Python package
datasets/                reviewed shareable dataset candidate
figures/                 generated figures for the report
model/                   reviewed trained-model candidate
results/                 lightweight aggregate results
```

Private XML/PDD files, occurrence arrays, spreadsheets, local environments and
record-level `CurrentECG` predictions must not be committed.
