from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modeling import extract_hybrid_metadata, positive_class_probability


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "Twitter_DataSET" / "bot_detection_data.csv"
OUTPUT_DIR = PROJECT_DIR / "ml_outputs"
RANDOM_STATE = 42


def _evaluate(model_name, model, features, labels, note):
    start = time.perf_counter()
    predictions = np.asarray(model.predict(features), dtype=int)
    bot_probabilities = positive_class_probability(model, features)
    elapsed = time.perf_counter() - start
    metrics = {
        "Model": model_name,
        "Dataset": "Twitter_DataSET bot_detection_data.csv",
        "Test samples": int(len(labels)),
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(labels, bot_probabilities) if bot_probabilities is not None else np.nan,
        "PR-AUC": average_precision_score(labels, bot_probabilities) if bot_probabilities is not None else np.nan,
        "Inference ms/sample": elapsed * 1000 / max(len(labels), 1),
        "Evaluation note": note,
    }
    print(f"\n{model_name}: accuracy={metrics['Accuracy']:.4f}, ROC-AUC={metrics['ROC-AUC']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}")
    print(classification_report(labels, predictions, target_names=["Human", "Bot"], zero_division=0))
    print("Confusion matrix [human, bot]:\n", confusion_matrix(labels, predictions, labels=[0, 1]))
    return metrics


def main():
    required_columns = [
        "Tweet", "Bot Label", "Retweet Count", "Follower Count", "Verified", "Created At",
    ]
    dataset = pd.read_csv(DATASET_PATH, usecols=required_columns)
    dataset = dataset.dropna(subset=["Tweet", "Bot Label"]).reset_index(drop=True)
    labels = pd.to_numeric(dataset["Bot Label"], errors="raise").astype(int).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Bot Label must contain both binary classes 0 and 1.")

    train_indices, test_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_tweets = dataset.iloc[train_indices]["Tweet"].fillna("").astype(str)
    test_tweets = dataset.iloc[test_indices]["Tweet"].fillna("").astype(str)
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    vectorizer = TfidfVectorizer(max_features=3000)
    text_train = vectorizer.fit_transform(train_tweets)
    text_test = vectorizer.transform(test_tweets)
    text_models = {
        "logistic_model_retrained.pkl": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest_model_retrained.pkl": RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }
    fitted_models = {}
    metric_rows = []
    note = "Stratified 80/20 holdout; all transforms fitted on training rows only; seed=42."
    for filename, model in text_models.items():
        print(f"Training {filename}...", flush=True)
        model.fit(text_train, train_labels)
        fitted_models[filename] = model
        model_name = "Logistic Regression (tweet text)" if filename.startswith("logistic") else "Random Forest (tweet text)"
        metric_rows.append(_evaluate(model_name, model, text_test, test_labels, note))

    reference_date = pd.Timestamp.now().normalize()
    metadata = extract_hybrid_metadata(dataset, reference_date=reference_date)
    scaler = StandardScaler()
    metadata_train = scaler.fit_transform(metadata.iloc[train_indices].to_numpy(dtype=float))
    metadata_test = scaler.transform(metadata.iloc[test_indices].to_numpy(dtype=float))
    hybrid_train = hstack([text_train, csr_matrix(metadata_train)], format="csr")
    hybrid_test = hstack([text_test, csr_matrix(metadata_test)], format="csr")
    hybrid_model = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=1,
    )
    print("Training Hybrid (tweet text + account metadata)...", flush=True)
    hybrid_model.fit(hybrid_train, train_labels)
    metric_rows.append(_evaluate(
        "Hybrid (tweet text + account metadata)", hybrid_model, hybrid_test, test_labels, note
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, model in fitted_models.items():
        joblib.dump(model, OUTPUT_DIR / filename)
    joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_retrained.pkl")
    joblib.dump(hybrid_model, OUTPUT_DIR / "hybrid_model_retrained.pkl")
    joblib.dump(vectorizer, OUTPUT_DIR / "hybrid_tfidf_retrained.pkl")
    joblib.dump(scaler, OUTPUT_DIR / "hybrid_scaler_retrained.pkl")
    (OUTPUT_DIR / "hybrid_reference_date.txt").write_text(reference_date.strftime("%Y-%m-%d"), encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    print(f"Saved matching text and hybrid artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
