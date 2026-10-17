# Research Evaluation Record

## Evaluation scope

This record captures results produced from the checked-in project artifacts and the Python 3.14.8 environment. Metrics are generated from actual predictions and labeled rows. They are not a claim that the system is ready for high-impact decisions.

## Label and feature checks

- `Twitter_DataSET/bot_detection_data.csv`: 50,000 rows; `Bot Label` counts are 0: 24,982 and 1: 25,018.
- The CSV contains 50,000 unique `User ID` values, no duplicate tweets, and no duplicate tweet strings with conflicting labels.
- The project conventions map 0 to Human and 1 to Bot. The CSV's upstream label-generation provenance is not included.
- Mean tweet length is 62.61 characters for label 0 and 62.64 for label 1. This check found no obvious text-length signal.
- The retrained text vectorizer has 969 learned terms; the text Logistic Regression and Random Forest each expect 969 features.
- The paired Hybrid model expects 978 features: 969 TF-IDF features plus nine metadata fields, with a matching nine-field scaler.
- The 100-record `TwiBot-20_sample.json` uses `domain` and has no explicit `label` property. `gcn_training.py` demonstrates a `"Bot" in domain` heuristic but does not write the checked-in 229,580-node graph tensors. The project lacks the preprocessing script and ID map that produced those tensors, so the full processed graph's original label lineage is not verifiable from this repository.

## Text and Hybrid holdout

The trainer uses a stratified 80/20 split with `random_state=42`. TF-IDF vocabulary and metadata scaling are fit on the training rows only. The evaluation rows are not used for fitting.

| Model | Test rows | Accuracy | Precision (Bot) | Recall (Bot) | F1 (Bot) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression (tweet text) | 10,000 | 0.5045 | 0.5049 | 0.5018 | 0.5034 | 0.5011 | 0.5012 |
| Random Forest (tweet text) | 10,000 | 0.5010 | 0.5013 | 0.5504 | 0.5247 | 0.5041 | 0.5051 |
| Hybrid (tweet text + account metadata) | 10,000 | 0.4961 | 0.4967 | 0.5240 | 0.5100 | 0.4989 | 0.5014 |

These results are near chance. Feature dimensions and the training/inference transforms match; label counts are balanced; duplicate/conflicting tweets were not found. The checked evidence does not identify a single implementation mismatch that explains the low scores. The remaining likely limitation is weak predictive signal or label/data quality in this CSV. Do not present the text or Hybrid models as reliable account detectors.

The profile Hybrid uses the saved text/metadata artifacts. Its retweet count is estimated from the entered engagement rate; this is an estimate, not fetched account data. Following count and posting rate contribute to the separately labeled heuristic score, not to the saved Hybrid's nine metadata fields.

## GraphSAGE protocol and results

`train_graphsage_research.py` performs a stratified 60/20/20 split of the 11,826 labeled graph nodes for each seed. Class weights are computed from training indices only. Training loss uses training labels only; validation loss selects the checkpoint with early stopping; the test set is evaluated once after selection. Graph edges and node features include the full graph during message passing, which is a transductive protocol. The completed run used seeds 42, 43, and 44, a maximum of 12 epochs, patience 4, learning rate 0.005, weight decay 0.0005, and the checked-in model architecture (8 input features, hidden width 256, 2 outputs). Exact arguments and checkpoint seed are saved in `ml_outputs/graphsage_research_run.json`.

| Seed | Best epoch | Test rows | Accuracy | Precision (Bot) | Recall (Bot) | F1 (Bot) | Macro F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 2,366 | 0.7844 | 0.7338 | 0.9621 | 0.8326 | 0.7650 | 0.7802 | 0.7478 |
| 43 | 12 | 2,366 | 0.7908 | 0.7413 | 0.9590 | 0.8363 | 0.7733 | 0.7926 | 0.7545 |
| 44 | 12 | 2,366 | 0.7202 | 0.7523 | 0.7420 | 0.7471 | 0.7170 | 0.7889 | 0.7555 |
| Mean ± sample SD | — | — | 0.7651 ± 0.0391 | 0.7425 ± 0.0093 | 0.8877 ± 0.1262 | 0.8053 ± 0.0504 | 0.7518 ± 0.0304 | 0.7872 ± 0.0064 | 0.7526 ± 0.0042 |

Seed-42 classification report:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Human | 0.9216 | 0.5611 | 0.6975 | 1,048 |
| Bot | 0.7338 | 0.9621 | 0.8326 | 1,318 |

Confusion matrix, rows = true Human/Bot and columns = predicted Human/Bot: `[[588, 460], [50, 1268]]`.

The old reported GraphSAGE result of 81.3% accuracy, 0.842 ROC-AUC, and 0.829 PR-AUC was not reproduced with the new protocol and must not be used as the current measured result. The new three-seed mean is 76.51% accuracy. The research checkpoint and exact seed-42 split metadata are saved separately as `graphsage_model_research.pt` and `graphsage_research_splits.json`; metrics, predictions, class reports, and run arguments are in adjacent `graphsage_seed_*` and `graphsage_research_run.json` files.

## GCN provenance

The existing `gcn_model.pt` checkpoint has no companion training split, epoch/selection record, or saved evaluation manifest. The app can load it and produce graph-node predictions, and it can compute diagnostic metrics on a reconstructed seed-42 split, but it cannot prove those nodes were excluded from checkpoint training. The UI marks those metrics diagnostic and does not treat them as a certified holdout or directly compare them as a research result.

## Remaining limitations

- Text and Hybrid models are near chance on the available CSV; improve the dataset/label quality and collect a stronger external evaluation set before using these predictions operationally.
- The graph setup is transductive and the three-seed GraphSAGE study is small; repeat with more seeds and a genuinely held-out graph or temporal split before making generalization claims.
- The GCN checkpoint lacks split provenance; retrain it with the saved validation/test protocol before comparing it to GraphSAGE.
- The local SQLite login is for single-machine use. Public deployment requires managed authentication and persistent storage.
- Browser-native download dialogs were not automated in the test environment. AppTest verified the registered PDF/CSV controls and PDF payload validity; it did not click through the browser's OS save dialog.

## Reproduce

From `E:\FakeInfluencerProject` after installing `requirements.txt`:

```powershell
python train_ml.py
python train_graphsage_research.py --seeds 42 43 44 --max-epochs 12 --patience 4
python -m streamlit run app.py
```
