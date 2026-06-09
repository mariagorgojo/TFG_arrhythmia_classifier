# Trained Model Availability

The final trained hybrid CNN checkpoint is intentionally ignored by Git:

```text
hybrid_cnn1d_train_val_test_model.pt
```

Share or upload the checkpoint only after the appropriate data-governance
review. An authorized collaborator can place the reviewed file in this folder
to run exploratory `CurrentECG` inference.

The older two-way prototype checkpoint may also be kept locally as:

```text
hybrid_cnn1d_model.pt
```

The CurrentECG5 + Optuna checkpoint is also generated locally and ignored by
Git:

```text
hybrid_cnn1d_currentecg5_optuna_model.pt
```
