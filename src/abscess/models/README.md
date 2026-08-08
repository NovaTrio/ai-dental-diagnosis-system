# Abscess module — trained models

Version-controlled model weights for the periapical-lesion pipeline. These are
committed so a fresh clone works immediately — **you do not need to retrain
after checking out on another machine.**

```
src/abscess/models/
├── tooth_segmentation/
│   └── rf_tooth_model.joblib      Random Forest tooth pixel classifier   (~62 MB)
├── lesion_segmentation/
│   └── rf_lesion_model.joblib     Random Forest lesion pixel classifier  (~83 MB)
└── lesion_detection/
    ├── rf_detector.joblib         Random Forest image-level detector
    ├── svm_detector.joblib        Linear SVM image-level detector
    └── scaler.joblib              StandardScaler for the detector features
```

## Why here and not in `data/`

`data/abscess/raw/` is gitignored in its entirety, so anything stored there is
lost on clone. Weights therefore live under `src/`, matching the convention
already used by `src/canalwidth/models/` and `src/working_length/models/`.

The split is deliberate:

| Artifact | Location | Tracked |
|---|---|---|
| Trained weights | `src/abscess/models/` | yes |
| Predicted masks, overlays, metrics, visualizations | `data/abscess/raw/<module>_output/` | no |

Outputs stay in `data/` because they are reproducible from the weights.

## Compression

All files are saved with `joblib.dump(..., compress=3)`, set via `MODEL_COMPRESS`
in each module. This shrinks the two Random Forests by ~75% (575 MB → 145 MB
total) and keeps every file under GitHub's 100 MB hard limit. Loading a
compressed forest costs about 2 s.

Compression does not change predictions. Verified against the pre-compression
models on the same input: the output masks were pixel-for-pixel identical
(0 differing pixels). Predicted probabilities can differ by ~4e-16 — float64
machine epsilon, caused by NumPy summation order changing with buffer
alignment, not by data loss — which is far too small to flip a decision.

Retraining rewrites these files in place, already compressed.

## Retraining

Only needed if you change the training data or hyperparameters. Run from
`src/abscess/Segmentation/`, in this order — later stages consume the earlier
models' masks as input features:

```bash
python tooth_segmentation.py     # -> tooth_segmentation/rf_tooth_model.joblib
python lesion_segmentation.py    # -> lesion_segmentation/rf_lesion_model.joblib
python lesion_detection.py       # -> lesion_detection/*.joblib
```

Training inputs live in `data/abscess/raw/` (`croun_crops/`, `tooth_masks/`,
`mask/`) and are **not** version-controlled, so retraining requires that dataset.
That is precisely why the weights are committed.
