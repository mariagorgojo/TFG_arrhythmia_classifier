# TFG Arrhythmia Classifier

Prototype for classifying four labeled arrhythmia types from single-lead ECG
occurrence records:

```text
AF
Asystole
Brady
VT
```

The current model is a hybrid 1D CNN. It combines:

- a 2,048-sample ECG waveform window;
- rhythm and marker features extracted from the same occurrence record.

`CurrentECG` is not treated as a diagnostic class. It is analyzed separately as
an unlabeled exploratory cohort.

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

The shareable training-dataset candidate is:

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

## Train The Hybrid Model

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

## Exploratory CurrentECG Inference

`CurrentECG` inference requires the private occurrence manifest and extracted
occurrence arrays. These files are not included in GitHub. The trained
checkpoint is also ignored by default and must be placed in `model/` after the
appropriate review.

Example:

```powershell
python -m arrhythmia_classifier.currentecg_inference --manifest-path D:\private_data\occurrence_dataset\occurrences_manifest.csv --model-path model\hybrid_cnn1d_model.pt --output-dir results\currentecg_exploratory
```

The output is exploratory only. The classifier has no confirmed NSR/normal or
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

## Repository Contents

```text
arrhythmia_classifier/   active Python package
datasets/                reviewed shareable dataset candidate
model/                   reviewed trained-model candidate
results/                 lightweight aggregate results
```

Private XML/PDD files, occurrence arrays, spreadsheets, local environments and
record-level `CurrentECG` predictions must not be committed.
