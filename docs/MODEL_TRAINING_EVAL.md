# Model Training and Evaluation

This section explains how `train/train.py` and `train/grid_search.py` actually turn Feast-served features into a scored, calibrated model, and reads through a real run's output (`*.metrics.json`, `*.report.txt`, `*.importance.json`) as a worked example.

### Splitting the data

`train/dataset.py` sorts every row by `TransactionDT` and cuts it chronologically — 70% train / 15% validation / 15% test (`time_based_split`) — instead of a random or k-fold split. Training only ever sees the past relative to validation and test, which mirrors how the production scorer will actually be used (scoring transactions that happen after the model was trained) and avoids leaking future information into training. `prepare_xy` then drops `isFraud`, `TransactionDT`, and `TransactionID`, keeps the numeric/boolean feature columns, and fills missing values with `0`.

### Handling class imbalance

Fraud is rare (roughly 1–4% positive depending on the time window), so `train_xgb_classifier` computes `scale_pos_weight` directly from the training split's class ratio (`negatives / positives`) and passes it to XGBoost instead of oversampling or undersampling rows. The model itself is `xgb.XGBClassifier` with `tree_method="hist"`, `eval_metric="aucpr"` (average precision, which is far more informative than accuracy at this base rate), and `early_stopping_rounds=30` monitored against the validation split.

### Hyperparameter search

`train/grid_search.py` calls `grid_search_xgb_classifier`, which searches `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, and `colsample_bytree` over a fixed grid. It deliberately does **not** use k-fold cross-validation: it carves the last `--search-validation-fraction` (default 20%) of the *training* split off chronologically as a throwaway search holdout, scores every candidate by average precision on that holdout, then refits the winning configuration on the complete training split and evaluates it on the project's real validation/test splits. Every candidate is logged as a child MLflow run; the winning config, metrics, and the full search table are logged on the parent run (see `make demo`'s sibling command in [3. Prepare features and train](#3-prepare-features-and-train) for how to browse these in the MLflow UI).

### Calibration

Raw XGBoost scores are not well-calibrated probabilities. `calibrate_classifier` wraps the trained booster in `sklearn.calibration.CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")` fit on the validation split (Platt scaling), so the resulting `fraud_prob` can be treated as an actual probability — the FastAPI `/predict` response returns it as-is, and the drift console's cost-weighting (`p_fraud` in `monitor/cost_drift.py`) depends on it being meaningful, not just rank-ordered.

### Choosing the decision threshold

`find_best_threshold` sweeps the validation precision-recall curve and picks the threshold that maximizes F-beta. `train/train.py` exposes this as `--threshold-beta` (default `1.0`, i.e. F1); `train/grid_search.py` evaluates with the library default of `beta=2.0`, which weighs recall twice as heavily as precision — a reasonable default here because missing a fraud (false negative) is modeled as far more expensive than investigating a false positive (see the `c_fn=$420` vs `c_fp=$4` defaults in the demo console). The threshold is tuned once on validation and then applied unchanged to the test split and to production — it is not re-tuned per request.

### What gets evaluated and saved

`evaluate_model` reports, on the held-out test split: average precision (`aucpr_test`), ROC-AUC (`roc_auc_test`, left `null` if the test slice happens to contain only one class), the predicted positive rate, and a full `sklearn.classification_report` at the chosen threshold. `feature_importance` separately pulls XGBoost's gain-based importance (total split gain per feature) from the uncalibrated booster. `save_bundle` then writes several files next to each other. Below, `<output>` is a placeholder for whatever path you passed to `--output` with its extension removed — for example, `--output outputs/model-grid-search.joblib` makes `<output>` equal to `outputs/model-grid-search`, so the files on disk are literally named `outputs/model-grid-search.metrics.json`, `outputs/model-grid-search.importance.json`, and so on:

- `<output>.joblib` — the artifact `app/model_loader.py` loads in production: calibrated model, feature list, threshold, metrics, importance, and the categorical-mapping fingerprint.
- `<output>.metrics.json` — the raw metrics dict shown below.
- `<output>.importance.json` — the gain-based importance dict (also what `demo/prepare_drift_windows.py` reads for the drift console's cost weighting).
- `<output>.report.txt` — the classification report in a flat, greppable text format.
- (grid search only) `<output>.grid-search.csv` and `<output>.grid-search-runs/config-NNN/{config.json,result.json}` — one row/folder per hyperparameter candidate tried.

None of this promotes the model automatically — `train/publish_champion.py` / `train/promote.py` copy a chosen `.joblib` bundle to the shared `gs://<PROJECT_ID>-artifacts/models/champion/model.joblib` location as an explicit, separate step (see [Step 1 — Publish the champion model to Cloud Storage](#step-1--publish-the-champion-model-to-cloud-storage)).

### A worked example (`model-grid-search.*`)

The numbers below are from one completed `python train/grid_search.py --output outputs/model-grid-search.joblib` run, read straight out of its `.metrics.json`, `.report.txt`, and `.importance.json` outputs.

**Search and model configuration**

| Setting | Value |
|---|---|
| Search rows (chronological holdout inside train) | 330,933 train / 82,734 validation |
| Best search average precision | 0.5405 |
| Selected hyperparameters | `n_estimators=500`, `max_depth=10`, `learning_rate=0.1`, `min_child_weight=3`, `subsample=0.85`, `colsample_bytree=0.8` |
| Decision threshold (tuned on validation, F2) | 0.0476 |
| Validation F-beta at that threshold | 0.5386 |

**Test-split performance** (88,644 rows, 3,092 of them fraud — a 3.49% base rate)

| Metric | Value |
|---|---|
| Average precision (PR-AUC) | 0.4732 |
| ROC-AUC | 0.8713 |
| Predicted positive rate | 4.84% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 — legitimate | 0.982 | 0.969 | 0.975 | 85,552 |
| 1 — fraud | 0.375 | 0.521 | 0.436 | 3,092 |
| **Accuracy** | | | **0.953** | 88,644 |
| Macro avg | 0.679 | 0.745 | 0.706 | 88,644 |
| Weighted avg | 0.961 | 0.953 | 0.957 | 88,644 |

**Top features by gain-based importance**

| Feature | Gain |
|---|---|
| `V70` | 380.0 |
| `V274` | 249.3 |
| `V315` | 169.4 |
| `V91` | 153.3 |
| `V265` | 133.3 |
| `V30` | 93.1 |
| `V312` | 83.3 |
| `merch_fraud_rate_30d` | 83.1 |
| `V264` | 82.3 |
| `card6__encoded` | 73.4 |

**How to read this run:**

- At this ~3.5% fraud base rate, average precision (0.47) and the per-class report are more informative than overall accuracy.
- The threshold is tuned low (4.8%) on purpose: the F2 objective favors recall, so the model catches about half of all fraud (52% recall) at a precision of 38% — a trade-off chosen deliberately, since the assumed cost of missing a fraud case is far higher than the cost of reviewing a false alarm.
- `merch_fraud_rate_30d` — one of the engineered Feast features (merchant-level rolling fraud rate), not a raw Kaggle column — ranks in the top 10 by importance, alongside the dataset's anonymized `V`/`card` columns. That's a good sign that the velocity/merchant-risk feature engineering in `ingest/build_features.py` and `feature_repo/features.py` is contributing real signal.
- `search_train_rows` / `search_validation_rows` describe the grid search's internal, throwaway chronological holdout (carved out of the training split only); they are separate from — and smaller than — the project's real validation/test splits that the reported test metrics above come from.