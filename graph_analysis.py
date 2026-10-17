from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from gcn_predict import load_graph_data


@lru_cache(maxsize=1)
def analyze_labeled_graph(base_dir: str | Path) -> dict[str, object]:
    graph_data = load_graph_data(base_dir)
    labels = graph_data["labels"].cpu().numpy().astype(int)
    edge_index = graph_data["edge_index"].cpu().numpy()
    labeled_count = len(labels)

    graph = nx.Graph()
    graph.add_nodes_from(range(labeled_count))
    valid = (edge_index[0] < labeled_count) & (edge_index[1] < labeled_count)
    labeled_edges = edge_index[:, valid].T
    graph.add_edges_from((int(source), int(target)) for source, target in labeled_edges)

    communities = nx.community.louvain_communities(graph, seed=42) if graph.number_of_edges() else [{node} for node in graph]
    node_community = {}
    rows = []
    for community_id, members in enumerate(communities):
        member_ids = sorted(int(node) for node in members)
        bot_ids = [node for node in member_ids if labels[node] == 1]
        human_count = len(member_ids) - len(bot_ids)
        subgraph = graph.subgraph(member_ids)
        bot_share = len(bot_ids) / len(member_ids) if member_ids else 0.0
        rows.append({
            "community_id": community_id,
            "nodes": len(member_ids),
            "edges": subgraph.number_of_edges(),
            "bot_accounts": len(bot_ids),
            "human_accounts": human_count,
            "bot_share": bot_share,
            "suspicious": len(member_ids) >= 5 and bot_share >= 0.6,
            "bot_node_ids": ",".join(map(str, bot_ids[:100])),
        })
        for node in member_ids:
            node_community[node] = community_id

    communities_df = pd.DataFrame(rows).sort_values(
        ["suspicious", "bot_share", "nodes"], ascending=[False, False, False]
    ).reset_index(drop=True)
    nodes_df = pd.DataFrame({
        "node_id": np.arange(labeled_count),
        "label": np.where(labels == 1, "Bot", "Human"),
        "degree": [graph.degree(node) for node in range(labeled_count)],
        "community_id": [node_community[node] for node in range(labeled_count)],
    })

    return {
        "graph": graph,
        "labels": labels,
        "nodes": nodes_df,
        "communities": communities_df,
        "edge_index": edge_index,
        "labeled_edges": labeled_edges,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }


def visual_subgraph(analysis: dict[str, object], node_limit: int = 300, edge_limit: int = 3000):
    labels = analysis["labels"]
    edge_index = analysis["labeled_edges"]
    node_limit = max(1, min(int(node_limit), len(labels)))
    selected_edges = edge_index[(edge_index[:, 0] < node_limit) & (edge_index[:, 1] < node_limit)]
    if edge_limit > 0 and len(selected_edges) > edge_limit:
        selected_edges = selected_edges[:edge_limit]
    return labels[:node_limit], selected_edges