# FictiPay Churn Prediction — bKash Presents NSUCEC Datathon

A gradient-boosting pipeline that predicts 30-day customer inactivity ("churn") for a mobile wallet, built for the bKash-sponsored NSUCEC Cybernauts Datathon on Kaggle.

## Overview

The task: given three months (Jan–Mar 2024) of raw transaction logs, daily balance snapshots, and KYC data for ~850K mobile wallet accounts, predict the probability that each customer will have **no transaction in April 2024**. Performance is scored by **ROC-AUC** on raw, continuous probabilities.

This notebook (`nsu-datathon-v2.ipynb`) is an iteration on an earlier baseline (OOF AUC 0.98531), adding survival/hazard-style features, a wider multi-source activity signal, and a more disciplined (OOF-safe) ensembling procedure.

## Dataset

| File | Description |
|---|---|
| `trx_2024-0{1,2,3}.parquet` | Per-transaction log: sender, receiver, timestamp, type, amount |
| `balance_2024-0{1,2,3}.parquet` | Daily end-of-day account balance panel |
| `kyc.parquet` | Customer KYC attributes (account type, open date, gender, region) |
| `train_labels.csv` | `ACCOUNT_ID` → `CHURN` (595K rows, 12.68% positive rate) |
| `test.csv` | `ACCOUNT_ID` to score (255K rows) |

All feature windows are pinned to **2024-04-01** as the prediction boundary and look strictly backward, so there is no leakage from the label period.

## Pipeline

**1. Feature engineering** (Polars, streaming/lazy scans to avoid materializing ~200M transaction rows)
- **Recency & counts**: transaction counts and amounts over 3/5/7/10/14/30/90-day windows, sent and received separately
- **Cadence / survival features**: mean gap, max gap, gap std/median/CV between active days, "unprecedented silence" (current silence exceeds historical max gap), Poisson-hazard churn probability
- **Multi-source activity**: days since last transaction, last incoming transfer, and last balance change, combined into a single "days since any activity" signal (a customer can look silent to the sender-side pipeline but still be active as a receiver)
- **Balance features**: mean/std/min balance, zero-balance days, balance-change frequency and recency
- **Interaction features**: transaction-vs-balance recency divergence, recency × gap-CV, inactive-days × balance-change
- **KYC**: account age, gender, region (smoothed, fold-safe target encoding)

Final matrix: **850,000 rows × 71 numeric features**.

**2. Class imbalance**: churn prevalence 12.68% → `scale_pos_weight ≈ 6.89` for the tree models.

**3. Modeling**: 5-fold stratified CV with LightGBM, XGBoost, and Logistic Regression (GPU-accelerated where available). CatBoost is implemented but gated off by default (a deeper CatBoost hurt an earlier version).

**4. Ensembling**: instead of fixed blend weights, non-negative ensemble weights are optimized directly against OOF AUC (SLSQP), with a safety net that falls back to the single best model if the optimized blend can't beat it.

**5. Explainability & leakage checks**: SHAP importance on a 5K sample, an assertion that no label/ID/raw-datetime columns leaked into the feature matrix, and a check that a recency/survival feature leads the importance ranking (sanity check for an inactivity label).

**6. Submission**: raw, unclipped, continuous probabilities — never thresholded to 0/1, since AUC-ROC is a ranking metric and collapsing to hard labels destroys the within-class ordering.

## Experiments (ablations, flag-gated)

| Flag | Idea | Result |
|---|---|---|
| `EXP1_DIVERGENCE` | Transaction-vs-balance/incoming recency divergence | On |
| `EXP2_CADENCE_EMP` | Empirical cadence: recency percentile in the gap CDF, recency × gap-CV | On |
| `EXP3_INACTIVE_IX` | inactive_days_30d × balance/incoming interactions | On |
| `EXP4_SPW_ABLATION` | Sweep `scale_pos_weight` over {1.0, √SPW, SPW} | Off by default |
| `EXP5_DECORR_ENS` | Seed-bagged LGBMs + LGBMRanker (lambdarank), rank-averaged | Partially worked — lambdarank hit a Kaggle GPU query-size limit and was skipped at runtime |

## Results

| Model | AUC | PR-AUC | Precision@10% | Recall@10% |
|---|---|---|---|---|
| LightGBM | 0.98464 | 0.9231 | 0.9138 | 0.7208 |
| XGBoost | **0.98507** | **0.9257** | **0.9165** | **0.7229** |
| Logistic Regression | 0.98413 | 0.9204 | 0.9089 | 0.7169 |
| Optimized ensemble | 0.98507 | 0.9257 | 0.9165 | 0.7229 |

The OOF weight optimizer collapsed to **XGBoost alone** (weight 1.0) — LightGBM and XGBoost turned out to be too highly correlated (pairwise OOF Pearson ~0.998) for blending to help, and the safety net correctly fell back to the single best model.

**Final ensemble OOF AUC: 0.98507**, a **-0.00024** delta versus the 0.98531 baseline this notebook builds on — i.e., this iteration's added features did not net-improve over the prior version on this run.

Top SHAP drivers: `txn_count_7d`, `inactive_x_bal_change`, `txn_count_last_14d`, `engagement_score`, `txn_count_last_10d` — a recency/survival feature leads, as expected for an inactivity label.

## Key lessons

- **Never submit thresholded labels.** Collapsing probabilities to 0/1 destroys the within-class ranking that ROC-AUC measures — this was the single biggest historical scoring bug, and fixing it produced a large leaderboard jump.
- **More features ≠ automatic improvement.** This iteration's added survival/cadence features slightly underperformed the earlier baseline on OOF AUC, a useful reminder to always compare against a fixed reference score rather than assuming added complexity helps.
- **Check ensemble member correlation before blending.** Near-duplicate models (LGBM/XGB here) leave the optimizer nothing to gain from blending; genuinely decorrelated members (e.g., a ranking objective) are needed for an ensemble to beat the best single model.
- **Multi-source recency matters.** A customer can be "sender-silent" but still active as a receiver or via balance movement — combining all three recency signals avoids false-positive churn flags.

## Repository structure

```
.
├── nsu-datathon-v2.ipynb   # Full pipeline: features → CV models → ensemble → submission
├── predictions.csv         # Output: ACCOUNT_ID, CHURN_PROB (raw probabilities)
├── features.md             # Auto-generated list of newly added features this iteration
└── explainability/
    └── shap_summary.png    # SHAP feature-importance bar chart
```

## How to run

Designed for a Kaggle notebook environment with the competition data mounted at `/kaggle/input/`.

```bash
pip install polars lightgbm xgboost scikit-learn shap optuna matplotlib pandas
```

Key toggles at the top of the notebook:
- `USE_GPU` — enable GPU training for LightGBM/XGBoost
- `USE_CATBOOST` — off by default (hurt performance in an earlier version)
- `EXP1`–`EXP5` — individual feature/ensemble experiments, flip one at a time and compare OOF AUC
- `RUN_OPTUNA` — off by default (features, not hyperparameters, were the bottleneck at this AUC level)

Running all cells top to bottom reproduces the feature matrix, 5-fold CV training, ensemble, SHAP explainability, and `predictions.csv`.
