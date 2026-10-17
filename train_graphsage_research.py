from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import SAGEConv

from gcn_predict import load_graph_data


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "ml_outputs"


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, out_channels: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, features, edge_index):
        features = self.dropout(F.relu(self.conv1(features, edge_index)))
        features = self.dropout(F.relu(self.conv2(features, edge_index)))
        return self.conv3(features, edge_index)


def split_indices(labels: np.ndarray, seed: int):
    all_indices = np.arange(len(labels))
    train_validation, test = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed,
        stratify=labels[train_validation],
    )
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        raise RuntimeError("Train, validation, and test indices overlap.")
    return train, validation, test


def train_one_seed(graph, seed: int, max_epochs: int, patience: int, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    features = graph["features"].to(device)
    edge_index = graph["edge_index"].to(device)
    labels = graph["labels"].cpu().numpy().astype(int)
    train_idx, validation_idx, test_idx = split_indices(labels, seed)
    classes = np.unique(labels[train_idx])
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Training split for seed {seed} does not contain both classes.")

    weights = compute_class_weight("balanced", classes=classes, y=labels[train_idx])
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    validation_tensor = torch.as_tensor(validation_idx, dtype=torch.long, device=device)
    test_tensor = torch.as_tensor(test_idx, dtype=torch.long, device=device)

    model = GraphSAGE(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(features, edge_index)
        loss = F.cross_entropy(logits[train_tensor], torch.as_tensor(labels[train_idx], device=device), weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_logits = model(features, edge_index)[validation_tensor]
            validation_loss = float(
                F.cross_entropy(
                    validation_logits,
                    torch.as_tensor(labels[validation_idx], device=device),
                ).item()
            )

        if validation_loss < best_validation_loss - 1e-4:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint was selected for seed {seed}.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        start = time.perf_counter()
        test_logits = model(features, edge_index)[test_tensor]
        probabilities = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        inference_seconds = time.perf_counter() - start

    predictions = (probabilities >= 0.5).astype(int)
    test_labels = labels[test_idx]
    metrics = {
        "seed": seed,
        "train_samples": len(train_idx),
        "validation_samples": len(validation_idx),
        "test_samples": len(test_idx),
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "accuracy": accuracy_score(test_labels, predictions),
        "precision_bot": precision_score(test_labels, predictions, zero_division=0),
        "recall_bot": recall_score(test_labels, predictions, zero_division=0),
        "f1_bot": f1_score(test_labels, predictions, zero_division=0),
        "f1_macro": f1_score(test_labels, predictions, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(test_labels, probabilities),
        "pr_auc": average_precision_score(test_labels, probabilities),
        "inference_ms_per_test_node": inference_seconds * 1000 / max(len(test_idx), 1),
        "confusion_matrix": confusion_matrix(test_labels, predictions, labels=[0, 1]).tolist(),
    }
    report = classification_report(
        test_labels,
        predictions,
        labels=[0, 1],
        target_names=["Human", "Bot"],
        output_dict=True,
        zero_division=0,
    )
    metrics["class_report"] = report
    prediction_rows = pd.DataFrame({
        "seed": seed,
        "node_index": test_idx,
        "true_label": test_labels,
        "predicted_label": predictions,
        "bot_probability": probabilities,
    })
    split = {
        "seed": seed,
        "train_indices": train_idx.tolist(),
        "validation_indices": validation_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    return model, metrics, prediction_rows, split


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Train GraphSAGE with train/validation/test splits and repeated seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.seeds or args.max_epochs < 1 or args.patience < 1:
        parser.error("seeds, max-epochs, and patience must be positive")

    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = load_graph_data(PROJECT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "status": "running",
        "seeds": args.seeds,
        "split_protocol": "Stratified 60/20/20 train/validation/test over labeled nodes",
        "class_weights_fit_on": "training labels only",
        "checkpoint_selected_by": "lowest validation cross-entropy",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rate": 0.005,
        "weight_decay": 5e-4,
        "hidden_channels": 256,
        "layers": 3,
        "input_features": int(graph["features"].shape[1]),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    manifest_path = OUTPUT_DIR / "graphsage_research_run.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    results = []
    predictions = []
    checkpoint_model = None
    checkpoint_split = None

    for seed in args.seeds:
        print(f"Training GraphSAGE seed={seed} on {device}...", flush=True)
        model, metrics, prediction_rows, split = train_one_seed(
            graph, seed, args.max_epochs, args.patience, device
        )
        results.append(metrics)
        predictions.append(prediction_rows)
        print(
            f"seed={seed} best_epoch={metrics['best_epoch']} "
            f"accuracy={metrics['accuracy']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
            f"PR-AUC={metrics['pr_auc']:.4f} F1(bot)={metrics['f1_bot']:.4f}"
        )
        if seed == 42 or checkpoint_model is None:
            checkpoint_model = model
            checkpoint_split = split
            checkpoint_split["seed"] = seed

    if checkpoint_model is not None:
        torch.save(checkpoint_model.state_dict(), OUTPUT_DIR / "graphsage_model_research.pt")
        (OUTPUT_DIR / "graphsage_research_splits.json").write_text(
            json.dumps(checkpoint_split), encoding="utf-8"
        )
    pd.DataFrame([{key: value for key, value in row.items() if key != "class_report" and key != "confusion_matrix"} for row in results]).to_csv(
        OUTPUT_DIR / "graphsage_seed_metrics.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT_DIR / "graphsage_seed_predictions.csv", index=False)
    (OUTPUT_DIR / "graphsage_seed_class_reports.json").write_text(
        json.dumps({str(row["seed"]): {"classification_report": row["class_report"], "confusion_matrix": row["confusion_matrix"]} for row in results}, indent=2),
        encoding="utf-8",
    )
    run_manifest.update({
        "status": "complete",
        "checkpoint_seed": checkpoint_split["seed"] if checkpoint_split else None,
        "best_epoch_by_seed": {str(row["seed"]): row["best_epoch"] for row in results},
    })
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    aggregate = pd.DataFrame(results)
    score_columns = ["accuracy", "precision_bot", "recall_bot", "f1_bot", "f1_macro", "roc_auc", "pr_auc"]
    print("\nRepeated-seed aggregate (mean +/- std):")
    for column in score_columns:
        print(f"{column}: {aggregate[column].mean():.4f} +/- {aggregate[column].std(ddof=1):.4f}")
    print(f"Saved research checkpoint, exact seed split, per-seed predictions, and reports to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()