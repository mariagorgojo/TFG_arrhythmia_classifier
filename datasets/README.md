# Dataset Availability

The training dataset is intentionally ignored by Git:

```text
ecg_training_dataset_arrhythmia4_features_deidentified.npz
```

The export removes original occurrence identifiers and replaces device
identifiers with generic groups, but the ECG waveforms remain clinical data.
Upload or share the file only after the appropriate data-governance review.

An authorized collaborator can place the reviewed `.npz` file in this folder
and run the training commands from the root `README.md`.
