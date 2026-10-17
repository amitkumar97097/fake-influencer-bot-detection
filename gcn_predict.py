from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_graph_data(project_dir: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for graph inference.") from exc

    data_dir = Path(project_dir) / "processed_data"
    filenames = ("edge_index.pt", "label.pt", "cat_properties_tensor.pt", "num_properties_tensor.pt")
    missing = [name for name in filenames if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Graph inference data is missing: {', '.join(missing)}")

    tensors = {
        name: torch.load(data_dir / name, map_location="cpu", weights_only=True)
        for name in filenames
    }
    edge_index = tensors["edge_index.pt"]
    labels = tensors["label.pt"].reshape(-1).long()
    cat_features = tensors["cat_properties_tensor.pt"]
    num_features = tensors["num_properties_tensor.pt"]
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}")
    if cat_features.ndim != 2 or num_features.ndim != 2:
        raise ValueError("Graph feature tensors must both be two-dimensional.")
    if cat_features.shape[0] != num_features.shape[0]:
        raise ValueError("Categorical and numeric graph feature tensors have different node counts.")
    if labels.numel() > cat_features.shape[0]:
        raise ValueError("More labels exist than graph feature rows.")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= cat_features.shape[0]):
        raise ValueError("edge_index contains node IDs outside the graph feature range.")

    features = torch.cat([cat_features.float(), num_features.float()], dim=1).contiguous()
    return {
        "edge_index": edge_index.long().contiguous(),
        "labels": labels,
        "cat_features": cat_features.float().contiguous(),
        "num_features": num_features.float().contiguous(),
        "features": features,
        "num_nodes": int(features.shape[0]),
        "num_labeled": int(labels.numel()),
    }


def load_graph_data(base_dir: str | Path = PROJECT_DIR) -> dict[str, Any]:
    """Load and validate graph tensors only when a graph feature is requested."""
    return _load_graph_data(str(Path(base_dir).resolve()))


@lru_cache(maxsize=2)
def _load_graph_model(model_name: str, project_dir: str):
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv, SAGEConv
    except ImportError as exc:
        raise RuntimeError("PyTorch and torch-geometric are required for graph inference.") from exc

    graph = load_graph_data(project_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized_name = model_name.casefold()

    if normalized_name == "gcn":
        class GCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(graph["features"].shape[1], 128)
                self.conv2 = GCNConv(128, 128)
                self.conv3 = GCNConv(128, 2)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, features, edge_index):
                features = self.dropout(F.relu(self.conv1(features, edge_index)))
                features = self.dropout(F.relu(self.conv2(features, edge_index)))
                return self.conv3(features, edge_index)

        model_class = GCN
        candidates = ["gcn_model.pt"]
    elif normalized_name in {"graphsage", "sage"}:
        class GraphSAGE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = SAGEConv(graph["features"].shape[1], 256)
                self.conv2 = SAGEConv(256, 256)
                self.conv3 = SAGEConv(256, 2)
                self.dropout = torch.nn.Dropout(p=0.5)

            def forward(self, features, edge_index):
                features = self.dropout(F.relu(self.conv1(features, edge_index)))
                features = self.dropout(F.relu(self.conv2(features, edge_index)))
                return self.conv3(features, edge_index)

        model_class = GraphSAGE
        candidates = ["graphsage_model_research.pt", "graphsage_model.pt"]
    else:
        raise ValueError("Graph model must be 'GCN' or 'GraphSAGE'.")

    model_dir = Path(project_dir) / "ml_outputs"
    model_path = next((model_dir / name for name in candidates if (model_dir / name).is_file()), None)
    if model_path is None:
        raise FileNotFoundError(f"{model_name} checkpoint not found in {model_dir}; expected one of: {', '.join(candidates)}")
    model = model_class().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"{model_name} checkpoint does not match the expected architecture: {exc}") from exc
    model.eval()
    return model, device


@lru_cache(maxsize=2)
def _predict_all_nodes_cached(model_name: str, project_dir: str):
    import torch

    graph = load_graph_data(project_dir)
    model, device = _load_graph_model(model_name, project_dir)
    features = graph["features"].to(device)
    edge_index = graph["edge_index"].to(device)
    with torch.inference_mode():
        logits = model(features, edge_index)
        probabilities = torch.softmax(logits[: graph["num_labeled"]], dim=1)
    return probabilities.cpu().numpy()


def predict_all_nodes(model_name: str, base_dir: str | Path = PROJECT_DIR):
    """Return cached probabilities for the labeled graph nodes."""
    project_dir = str(Path(base_dir).resolve())
    return _predict_all_nodes_cached(model_name.casefold(), project_dir).copy()


def predict_graph_user(user_index: int, model_name: str = "GCN", base_dir: str | Path = PROJECT_DIR):
    """Predict one labeled graph user with a verified local checkpoint."""
    graph = load_graph_data(base_dir)
    index = int(user_index)
    if not 0 <= index < graph["num_labeled"]:
        raise IndexError(f"User index must be between 0 and {graph['num_labeled'] - 1}.")

    probabilities = predict_all_nodes(model_name, base_dir)[index]
    prediction = int(probabilities.argmax())
    confidence = float(probabilities[prediction])
    bot_probability = float(probabilities[1])

    numeric = graph["num_features"][index].cpu().tolist()
    degree = int(((graph["edge_index"][0] == index) | (graph["edge_index"][1] == index)).sum().item())
    user_info = {
        "followers": round(float(numeric[0])) if len(numeric) > 0 else 0,
        "following": round(float(numeric[1])) if len(numeric) > 1 else 0,
        "tweets": round(float(numeric[2])) if len(numeric) > 2 else 0,
        "account_age": round(float(numeric[3])) if len(numeric) > 3 else 0,
        "engagement_score": round(float(numeric[4]), 2) if len(numeric) > 4 else 0.0,
        "graph_degree": degree,
    }
    return {
        "prediction": prediction,
        "confidence": confidence,
        "bot_probability": bot_probability,
        "true_label": int(graph["labels"][index].item()),
        "user_info": user_info,
    }


def predict_user(user_index: int):
    """Backward-compatible GCN API used by existing app code."""
    result = predict_graph_user(user_index, "GCN")
    return result["prediction"], result["confidence"], result["true_label"], result["user_info"]