# TFG Arrhythmia Classifier

Machine-learning pipeline for classifying arrhythmia-related ECG occurrence
records exported from Medtronic Reveal LINQ and LINQ II insertable cardiac
monitors.

This repository supports the final degree project:
Development of a Machine Learning-based arrhythmia classifier using data from
Medtronic LINQ insertable cardiac monitors


The active thesis task uses five supervised labels:

- CurrentECG
- AF
- Asystole
- Brady
- VT

CurrentECG is preserved as the original device/export label. In this project
it is treated operationally as the baseline class, but it is not renamed as
clinically confirmed normal sinus rhythm.

## Results

The thesis draft reports a hybrid CNN-BiLSTM as the selected final model and a
150-trial hybrid CNN as the convolutional baseline.

This repository includes aggregate result artifacts. The source clinical data
used to derive the training dataset are not included.

|Result folder |Role |
|---|---|
| results/hybrid_cnn1d_bilstm_optuna/ | Selected CNN-BiLSTM test result artifacts. |
| results/hybrid_cnn1d_currentecg5_optuna_150trials/ | 150-trial hybrid CNN baseline artifacts. |
| results/hybrid_cnn1d_currentecg5_optuna_100trials/ | Earlier 100-trial hybrid CNN run; historical/intermediate. |

Final test-set performance reported in the thesis:

| Metric | Hybrid CNN baseline | CNN-BiLSTM selected model |
|---|---:|---:|
| Accuracy | 0.991 | 0.993 |
| Macro F1 | 0.966 | 0.980 |
| Weighted F1 | 0.991 | 0.993 |
| Arrhythmia-only macro F1 | 0.958 | 0.975 |

Per-class performance of the selected CNN-BiLSTM:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| CurrentECG | 1.000 | 0.996 | 0.998 | 16,866 |
| AF | 0.922 | 0.998 | 0.959 | 1,033 |
| Asystole | 0.989 | 0.984 | 0.986 | 2,036 |
| Brady | 0.965 | 0.977 | 0.971 | 784 |
| VT | 0.994 | 0.975 | 0.984 | 829 |

The main remaining errors in the selected model are:

- Asystole predicted as Brady: 28 records.
- Brady predicted as Asystole: 16 records.
- CurrentECG or VT records over-called as AF: 85 records.

## Repository Structure And Reproducibility

### arrhythmia_classifier/

| File | Role |
|---|---|
| cnn1d_classifier.py | Historical waveform-only CNN baseline. |
| hybrid_cnn1d_classifier.py | Historical hybrid CNN model using ECG waveform and rhythm features. |
| hybrid_cnn1d_train_val_test.py | Device-disjoint train/validation/test splitting and evaluation helpers. |
| hybrid_cnn1d_currentecg5_optuna.py | Five-class Optuna hybrid CNN baseline. |
| hybrid_cnn_bilstm_optuna.py | Five-class CNN-BiLSTM selected model. |
| plot_ecg_examples.py | Generates representative ECG example plots. |

### dataset/

| File | Role |
|---|---|
| dataset_builder.py | Extracts occurrence-level records from Medtronic XML exports. |
| dataset_loader.py | Builds supervised ECG datasets from extracted occurrence records. |
| export_pseudonymized_currentecg5_dataset.py | Exports the final five-class dataset with generic device groups and removed occurrence IDs. |

### results/

| Folder | Role |
|---|---|
| hybrid_cnn1d_bilstm_optuna/ | Selected CNN-BiLSTM aggregate result. |
| hybrid_cnn1d_currentecg5_optuna_150trials/ | Hybrid CNN baseline aggregate result. |
| hybrid_cnn1d_currentecg5_optuna_100trials/ | Earlier intermediate 100-trial run. |

### tests/

| File | Role |
|---|---|
| test_dataset_builder.py | Synthetic XML extraction test. |

The final models require the five-class ECG dataset. A pseudonymized dataset
export helper is included to support controlled sharing of the derived dataset.
The pseudonymized dataset remains clinical signal data because it contains ECG
waveforms and rhythm-derived features.

Raw XML/PDD exports, patient spreadsheets, full occurrence arrays, derived ECG
datasets and trained model checkpoints are excluded from the public repository
unless explicitly reviewed for sharing.

## Data Pipeline

The XML processing pipeline is:

1. Extract cardiac occurrences from Medtronic XML exports.
2. Store each occurrence waveform as a compressed NumPy array.
3. Write an occurrence manifest with metadata, device group and label.
4. Filter usable records to 128 Hz stored ECG waveforms.
5. Encode each ECG as a fixed 2,048-sample window.
6. Compute rhythm-derived features from device-exported beat markers.
7. Split train, validation and test sets by device to avoid leakage.

The final supervised dataset contains 104,261 records:

| Class | Records |
|---|---:|
| CurrentECG | 82,668 |
| Asystole | 9,725 |
| AF | 5,043 |
| VT | 4,880 |
| Brady | 1,945 |

The dataset is not included in the repository because it is derived from
clinical device exports.

## Installation

This repository has two requirement files:

| File | Purpose |
|---|---|
| requirements.txt | Runtime dependencies for training and analysis. |
| requirements-dev.txt | Runtime dependencies plus `pytest` for tests. |

Create and activate a Python environment:

| Environment | Command |
|---|---|
| macOS/Linux | `python3 -m venv .venv` |
| macOS/Linux | `source .venv/bin/activate` |
| macOS/Linux | `python -m pip install --upgrade pip` |
| macOS/Linux | `python -m pip install -r requirements.txt` |

On Windows PowerShell:

| Environment | Command |
|---|---|
| Windows PowerShell | `python -m venv .venv` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows PowerShell | `python -m pip install --upgrade pip` |
| Windows PowerShell | `python -m pip install -r requirements.txt` |

Run the tests:

| Step | Command |
|---|---|
| Install test dependencies | `python -m pip install -r requirements-dev.txt` |
| Run tests | `python -m pytest` |

## Rebuild Datasets From Private XML Files

Raw XML/PDD files are intentionally excluded from the repository. With an
authorized local XML folder, rebuild occurrence arrays with:

| Step | Command |
|---|---|
| Extract XML occurrences | `python -m arrhythmia_classifier.dataset_builder --xml-dir /path/to/private/xml_exports --output-dir /path/to/private/occurrence_dataset` |

Build the current five-class dataset:

| Step | Command |
|---|---|
| Build five-class dataset | `python -m arrhythmia_classifier.dataset_loader --manifest-path /path/to/private/occurrence_dataset/occurrences_manifest.csv --label-strategy clinical5 --output-path /path/to/private/ecg_training_dataset_currentecg5_features.npz --max-per-label 0 --selection-strategy device_round_robin` |

## Train The Hybrid CNN Baseline

The code can reproduce the hybrid CNN baseline, provided the five-class dataset
is available locally:

| Step | Command |
|---|---|
| Train hybrid CNN baseline | `python -m arrhythmia_classifier.hybrid_cnn1d_currentecg5_optuna --dataset-path /path/to/private/ecg_training_dataset_currentecg5_features.npz --output-dir results/hybrid_cnn1d_currentecg5_optuna_150trials --n-trials 150 --min-epochs 6 --max-epochs 20 --patience 5` |

## Tests

The tests are intentionally small and do not require clinical data. They create
a synthetic Medtronic-like XML file and verify that the extraction code
preserves waveform sampling, amplitude scale, marker count and manifest output:

| File | Role |
|---|---|
| tests/conftest.py | Adds the repository root to the Python path for tests. |
| tests/test_dataset_builder.py | Tests extraction from a synthetic Medtronic-like XML file. |

## Notes For Thesis Interpretation

Accuracy and weighted F1 are high partly because `CurrentECG` is the majority
class and is almost perfectly separated. Macro F1 and arrhythmia-only macro F1
are therefore more informative for comparing rhythm discrimination.

The CNN-BiLSTM improves the slow-rhythm boundary between Asystole and Brady
relative to the hybrid CNN baseline. AF has very high recall but lower precision
because the selected model slightly over-predicts AF.

This project is an academic prototype. It is not a clinical diagnostic system
and would require clinical review, external validation and regulatory assessment
before any clinical use.
