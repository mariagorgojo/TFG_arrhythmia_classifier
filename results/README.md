# Aggregate Results

Only lightweight aggregate outputs are stored here. Record-level predictions,
raw XML/PDD files and occurrence arrays are intentionally excluded.

## Validated Test Set

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Waveform-only CNN 1D | 0.6685 | 0.6491 | 0.6814 |
| Hybrid CNN 1D | 0.7863 | 0.7700 | 0.8090 |

The hybrid model combines ECG morphology with rhythm and marker features.

## Exploratory CurrentECG Cohort

`CurrentECG` does not provide confirmed diagnostic labels. Its outputs are
exploratory forced-choice predictions, not validated diagnoses.

| Predicted output | Records | Percentage |
|---|---:|---:|
| AF | 73,992 | 89.51% |
| Asystole | 8,659 | 10.47% |
| Brady | 3 | <0.01% |
| VT | 14 | 0.02% |

The model has no confirmed NSR/normal or other/indeterminate output. The strong
AF concentration must not be interpreted as clinical prevalence.

