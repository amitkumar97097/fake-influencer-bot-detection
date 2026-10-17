# Fake Influencer / Bot Detection System

A local Streamlit application for tweet-text classification, profile heuristics and trained Hybrid scores, graph-node inference, graph exploration, and holdout evaluation.

## Windows setup

Use the same 64-bit Python 3.14 environment used to verify this project (currently Python 3.14.8). From PowerShell:

```powershell
Set-Location E:\FakeInfluencerProject
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m py_compile app.py auth.py modeling.py evaluation.py gcn_predict.py graph_analysis.py train_ml.py train_graphsage_research.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again.

Run the app:

```powershell
python -m streamlit run app.py
```

On first launch, create a local account from the sign-in screen. Passwords are stored as salted PBKDF2 hashes in `prediction_history.db`; prediction history and settings are scoped to that username. Do not expose this local SQLite authentication to a public service.

## Models and data

The app expects `Twitter_DataSET/bot_detection_data.csv`, the model artifacts under `ml_outputs`, and the graph tensors under `processed_data`. Text/Hybrid models can be rebuilt from the labeled tweet CSV with:

```powershell
python train_ml.py
```

Training uses a stratified 80/20 holdout, fits TF-IDF and metadata scaling on training rows only, and writes matching versioned artifacts plus measured text/Hybrid metrics to `ml_outputs/model_summary.csv`. Do not interpret weak measured accuracy as a reliable production detector.

GCN and GraphSAGE inference require their matching `.pt` checkpoints and all four graph tensors. Missing files, incompatible dimensions, or load errors are reported in the UI; the app does not fabricate graph predictions. The available Hybrid checkpoint is text plus account metadata, not a saved graph-embedding Hybrid checkpoint.

To retrain and evaluate GraphSAGE without using test labels for class weights or model selection, run:

```powershell
python train_graphsage_research.py --seeds 42 43 44 --max-epochs 12 --patience 4
```

This writes a separate `graphsage_model_research.pt`, exact seed-42 split indices, per-seed predictions/classification reports, `graphsage_seed_metrics.csv`, and `graphsage_research_run.json`. It does not overwrite the legacy GraphSAGE checkpoint.

## Verified research results

The tweet CSV contains 50,000 rows, labels `{0: 24,982, 1: 25,018}`, 50,000 distinct user IDs, no duplicate tweets, and no duplicate tweet strings with conflicting labels. The scripts map `0` to Human and `1` to Bot. Mean tweet lengths are nearly identical for the two labels (62.61 vs 62.64 characters). These checks found no obvious label-count or feature-width mismatch, but cannot establish the dataset's label provenance or semantic quality.

The current text-only and text-plus-metadata Hybrid artifacts were trained with a stratified 80/20 split (seed 42), fitting TF-IDF and metadata scaling on training rows only. On the 10,000-row holdout:

| Model | Accuracy | Precision (Bot) | Recall (Bot) | F1 (Bot) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression (text) | 0.5045 | 0.5049 | 0.5018 | 0.5034 | 0.5011 | 0.5012 |
| Random Forest (text) | 0.5010 | 0.5013 | 0.5504 | 0.5247 | 0.5041 | 0.5051 |
| Hybrid (text + metadata) | 0.4998 | 0.5002 | 0.5392 | 0.5189 | 0.5034 | 0.5055 |

These text/Hybrid scores are near chance. Investigation found a concrete previous Hybrid feature mismatch: the CSV `Hashtags` and `Mention Count` columns matched counts parsed from `Tweet` only 16.68% and 16.46% of the time over all 50,000 rows. The corrected trainer now derives both counts from raw tweet text, the same parser used at inference; TF-IDF and metadata scaling are fit on training rows only. The corrected Hybrid score remains near chance (49.98%), so this mismatch was not the only limitation. Label semantics and source provenance remain unverified. Do not use these scores as reliable bot detection.

GraphSAGE was retrained with a stratified 60/20/20 train/validation/test split for seeds 42, 43, and 44. Class weights are computed from training labels only; validation loss selects the checkpoint; the test split is not used for training or early stopping. All three use the same 2,366-node test size:

| Seed | Best epoch | Test rows | Accuracy | Precision (Bot) | Recall (Bot) | F1 (Bot) | Macro F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 2,366 | 0.7844 | 0.7338 | 0.9621 | 0.8326 | 0.7650 | 0.7802 | 0.7478 |
| 43 | 12 | 2,366 | 0.7908 | 0.7413 | 0.9590 | 0.8363 | 0.7733 | 0.7926 | 0.7545 |
| 44 | 12 | 2,366 | 0.7202 | 0.7523 | 0.7420 | 0.7471 | 0.7170 | 0.7889 | 0.7555 |
| Mean ± sample SD | — | — | 0.7651 ± 0.0391 | 0.7425 ± 0.0093 | 0.8877 ± 0.1262 | 0.8053 ± 0.0504 | 0.7518 ± 0.0304 | 0.7872 ± 0.0064 | 0.7526 ± 0.0042 |

Seed-42 class-wise report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Human | 0.9216 | 0.5611 | 0.6975 | 1,048 |
| Bot | 0.7338 | 0.9621 | 0.8326 | 1,318 |

Seed-42 confusion matrix (rows = true Human/Bot; columns = predicted Human/Bot): `[[588, 460], [50, 1268]]`.

The earlier reported GraphSAGE result of 81.3% accuracy / 0.842 ROC-AUC / 0.829 PR-AUC was not reproduced under this verified protocol. The new three-seed mean is 76.51% accuracy; do not substitute the old figure for these results. Three seeds are preliminary, not a broad statistical guarantee.

## Evaluation and limitations

The Research Evaluation page computes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, inference time, and CSV exports from actual predictions against labeled rows. Text and Hybrid scores use the CSV tweet holdout; graph scores use labeled TwiBot nodes, so these are different datasets/tasks and are not directly comparable. The GCN checkpoint has no saved split metadata and is explicitly labeled diagnostic. Graph metrics are transductive: all graph edges/features are available to message passing while only train labels supervise loss; this is not an inductive unseen-graph test. The GraphSAGE research checkpoint and exact seed-42 indices are saved separately from the legacy checkpoint.

The checked-in TwiBot JSON sample has 100 records and does not contain an explicit `label` field. `gcn_training.py` demonstrates a `"Bot" in domain` heuristic but does not save the processed 229,580-node tensors. The project does not contain the exact preprocessing script or user-ID mapping that produced the checked-in tensors, so the full graph labels' original lineage cannot be independently verified. Treat graph results as conditional on those supplied processed labels.

## Deployment

For Streamlit Community Cloud, select `app.py` as the entry point and install from `requirements.txt`; include the required model and data artifacts in the deployment or fetch them from controlled storage at startup. Configure the hosted service's bind address for its platform. The local SQLite database is ephemeral on many hosted platforms and is not suitable for durable multi-user history. Before public deployment, replace local login with an identity provider (OIDC) and move user/history storage to a persistent managed database. Never commit credentials or secrets.
