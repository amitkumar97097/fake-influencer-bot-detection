from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from gcn_predict import load_graph_data, predict_all_nodes
from modeling import (
    extract_hybrid_metadata,
    load_model_resources,
    positive_class_probability,
    transform_hybrid,
)


def _result_row(name, dataset_name, labels, predictions, bot_probability, elapsed, note):
    probability = np.asarray(bot_probability, dtype=float) if bot_probability is not None else None
    return {
        "Model": name,
        "Dataset": dataset_name,
        "Test samples": int(len(labels)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "Precision": float(precision_score(labels, predictions, zero_division=0)),
        "Recall": float(recall_score(labels, predictions, zero_division=0)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(labels, probability)) if probability is not None else np.nan,
        "PR-AUC": float(average_precision_score(labels, probability)) if probability is not None else np.nan,
        "Inference ms/sample": float(elapsed * 1000 / max(len(labels), 1)),
        "Evaluation note": note,
    }


def _add_predictions(model_name, dataset_name, row_ids, labels, predictions, bot_probability, output_rows):
    probabilities = bot_probability if bot_probability is not None else [np.nan] * len(labels)
    output_rows.extend(
        {
            "Model": model_name,
            "Dataset": dataset_name,
            "sample_id": str(row_id),
            "true_label": int(label),
            "predicted_label": int(prediction),
            "bot_probability": float(probability) if np.isfinite(probability) else np.nan,
        }
        for row_id, label, prediction, probability in zip(row_ids, labels, predictions, probabilities)
    )


def _class_report(labels, predictions):
    return classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Human", "Bot"],
        output_dict=True,
        zero_division=0,
    )


def evaluate_text_models(project_dir: str | Path) -> dict[str, Any]:
    project_dir = Path(project_dir).resolve()
    dataset_path = project_dir / "Twitter_DataSET" / "bot_detection_data.csv"
    result = {"metrics": [], "predictions": [], "confusion_matrices": {}, "classification_reports": {}, "curves": {}, "errors": []}
    if not dataset_path.is_file():
        result["errors"].append(f"Tweet evaluation dataset not found: {dataset_path}")
        return result

    required = ["User ID", "Tweet", "Bot Label", "Retweet Count", "Mention Count", "Follower Count", "Verified", "Created At", "Hashtags"]
    try:
        dataset = pd.read_csv(dataset_path, usecols=required).dropna(subset=["Tweet", "Bot Label"]).reset_index(drop=True)
        labels = pd.to_numeric(dataset["Bot Label"], errors="raise").astype(int).to_numpy()
        train_ids, test_ids = train_test_split(
            np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=labels
        )
    except Exception as exc:
        result["errors"].append(f"Could not prepare the text holdout set: {type(exc).__name__}: {exc}")
        return result

    resources = load_model_resources(project_dir / "ml_outputs")
    text_vectorizer = resources.get("tfidf")
    if text_vectorizer is None:
        result["errors"].append("A matching TF-IDF artifact is unavailable; text-model metrics were skipped.")
        return result

    test_text = dataset.iloc[test_ids]["Tweet"].fillna("").astype(str).tolist()
    text_vectors = text_vectorizer.transform(test_text)
    test_labels = labels[test_ids]
    note = "Recomputed from the trainer's stratified 80/20 split (random_state=42)."
    text_names = {"lr_model": "Logistic Regression (tweet text)", "rf_model": "Random Forest (tweet text)"}
    for resource_name, model_name in text_names.items():
        model = resources.get(resource_name)
        if model is None:
            detail = resources.get("load_errors", {}).get(resource_name, "matching model artifact not found")
            result["errors"].append(f"{model_name}: {detail}")
            continue
        try:
            start = time.perf_counter()
            predictions = np.asarray(model.predict(text_vectors), dtype=int)
            probabilities = positive_class_probability(model, text_vectors)
            elapsed = time.perf_counter() - start
            row = _result_row(model_name, "Twitter_DataSET tweet holdout", test_labels, predictions, probabilities, elapsed, note)
            result["metrics"].append(row)
            result["confusion_matrices"][model_name] = confusion_matrix(test_labels, predictions, labels=[0, 1])
            result["classification_reports"][model_name] = _class_report(test_labels, predictions)
            _add_predictions(model_name, "Twitter_DataSET tweet holdout", dataset.iloc[test_ids]["User ID"].tolist(), test_labels, predictions, probabilities, result["predictions"])
            if probabilities is not None:
                result["curves"][model_name] = {
                    "roc": roc_curve(test_labels, probabilities),
                    "pr": precision_recall_curve(test_labels, probabilities),
                }
        except Exception as exc:
            result["errors"].append(f"{model_name} evaluation failed: {type(exc).__name__}: {exc}")

    hybrid_model = resources.get("hybrid_model")
    hybrid_vectorizer = resources.get("hybrid_tfidf")
    hybrid_scaler = resources.get("hybrid_scaler")
    if all(value is None for value in (hybrid_model, hybrid_vectorizer, hybrid_scaler)):
        result["errors"].append("Hybrid model, TF-IDF, and scaler artifacts are unavailable; Hybrid metrics were skipped.")
    elif not all(value is not None for value in (hybrid_model, hybrid_vectorizer, hybrid_scaler)):
        result["errors"].append("Hybrid artifacts are incomplete; a matching model/vectorizer/scaler trio is required.")
    else:
        try:
            reference_date = resources.get("hybrid_reference") or None
            metadata = extract_hybrid_metadata(dataset, reference_date=reference_date)
            features = transform_hybrid(
                dataset.iloc[test_ids]["Tweet"].fillna("").astype(str).tolist(),
                metadata.iloc[test_ids],
                hybrid_vectorizer,
                hybrid_scaler,
            )
            start = time.perf_counter()
            predictions = np.asarray(hybrid_model.predict(features), dtype=int)
            probabilities = positive_class_probability(hybrid_model, features)
            elapsed = time.perf_counter() - start
            model_name = "Hybrid (tweet text + account metadata)"
            row = _result_row(model_name, "Twitter_DataSET tweet holdout", test_labels, predictions, probabilities, elapsed, note)
            result["metrics"].append(row)
            result["confusion_matrices"][model_name] = confusion_matrix(test_labels, predictions, labels=[0, 1])
            result["classification_reports"][model_name] = _class_report(test_labels, predictions)
            _add_predictions(model_name, "Twitter_DataSET tweet holdout", dataset.iloc[test_ids]["User ID"].tolist(), test_labels, predictions, probabilities, result["predictions"])
            if probabilities is not None:
                result["curves"][model_name] = {
                    "roc": roc_curve(test_labels, probabilities),
                    "pr": precision_recall_curve(test_labels, probabilities),
                }
        except Exception as exc:
            result["errors"].append(f"Hybrid evaluation failed: {type(exc).__name__}: {exc}")
    return result


def evaluate_graph_models(project_dir: str | Path) -> dict[str, Any]:
    import torch

    project_dir = Path(project_dir).resolve()
    result = {"metrics": [], "predictions": [], "confusion_matrices": {}, "classification_reports": {}, "curves": {}, "errors": [], "cross_seed_metrics": None}
    try:
        graph = load_graph_data(project_dir)
    except Exception as exc:
        result["errors"].append(f"Graph evaluation unavailable: {type(exc).__name__}: {exc}")
        return result

    labels = graph["labels"].cpu().numpy().astype(int)
    generator = torch.Generator(device="cpu").manual_seed(42)
    permutation = torch.randperm(len(labels), generator=generator).numpy()
    legacy_test_ids = permutation[int(0.8 * len(labels)):]
    split_path = project_dir / "ml_outputs" / "graphsage_research_splits.json"
    research_test_ids = None
    if split_path.is_file():
        try:
            split_metadata = pd.read_json(split_path, typ="series").to_dict()
            candidates = np.asarray(split_metadata["test_indices"], dtype=int)
            if (
                len(candidates)
                and len(np.unique(candidates)) == len(candidates)
                and candidates.min() >= 0
                and candidates.max() < len(labels)
            ):
                research_test_ids = candidates
        except Exception as exc:
            result["errors"].append(f"GraphSAGE split metadata is invalid: {type(exc).__name__}: {exc}")
    seed_metrics_path = project_dir / "ml_outputs" / "graphsage_seed_metrics.csv"
    if seed_metrics_path.is_file():
        try:
            result["cross_seed_metrics"] = pd.read_csv(seed_metrics_path)
        except Exception as exc:
            result["errors"].append(f"GraphSAGE repeated-seed metrics could not be read: {type(exc).__name__}: {exc}")

    for model_name in ("GCN", "GraphSAGE"):
        try:
            test_ids = research_test_ids if model_name == "GraphSAGE" and research_test_ids is not None else legacy_test_ids
            test_labels = labels[test_ids]
            start = time.perf_counter()
            all_probabilities = predict_all_nodes(model_name, project_dir)
            elapsed = time.perf_counter() - start
            bot_probabilities = all_probabilities[test_ids, 1]
            predictions = all_probabilities[test_ids].argmax(axis=1).astype(int)
            note = "Test indices reconstructed with the graph trainer's seed-42 80/20 split."
            if model_name == "GCN":
                note += " Checkpoint training split was not saved; treat this row as diagnostic, not a certified holdout."
            elif research_test_ids is not None:
                note = "Test indices loaded from the saved stratified 60/20/20 research split; checkpoint selected by validation loss."
            else:
                note += " Legacy checkpoint has no saved validation/test metadata; this row is diagnostic."
            row = _result_row(
                f"{model_name} (graph)", "processed_data labeled-node holdout", test_labels,
                predictions, bot_probabilities, elapsed, note,
            )
            result["metrics"].append(row)
            result["confusion_matrices"][f"{model_name} (graph)"] = confusion_matrix(test_labels, predictions, labels=[0, 1])
            result["classification_reports"][f"{model_name} (graph)"] = _class_report(test_labels, predictions)
            _add_predictions(
                f"{model_name} (graph)", "processed_data labeled-node holdout", test_ids,
                test_labels, predictions, bot_probabilities, result["predictions"],
            )
            result["curves"][f"{model_name} (graph)"] = {
                "roc": roc_curve(test_labels, bot_probabilities),
                "pr": precision_recall_curve(test_labels, bot_probabilities),
            }
        except Exception as exc:
            result["errors"].append(f"{model_name} evaluation unavailable: {type(exc).__name__}: {exc}")
    return result


def evaluate_all_models(project_dir: str | Path) -> dict[str, Any]:
    text_results = evaluate_text_models(project_dir)
    graph_results = evaluate_graph_models(project_dir)
    return {
        "metrics": pd.DataFrame(text_results["metrics"] + graph_results["metrics"]),
        "predictions": pd.DataFrame(text_results["predictions"] + graph_results["predictions"]),
        "confusion_matrices": {**text_results["confusion_matrices"], **graph_results["confusion_matrices"]},
        "classification_reports": {**text_results["classification_reports"], **graph_results["classification_reports"]},
        "curves": {**text_results["curves"], **graph_results["curves"]},
        "cross_seed_metrics": graph_results["cross_seed_metrics"],
        "errors": text_results["errors"] + graph_results["errors"],
    }