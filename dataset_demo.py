import torch
import os

print("🔍 Loading TwiBot-20 Dataset Overview...\n")

DATA_PATH = "processed_data"

# ================= LOAD DATA =================
edge_index = torch.load(os.path.join(DATA_PATH, "edge_index.pt"))
labels = torch.load(os.path.join(DATA_PATH, "label.pt"))
cat_features = torch.load(os.path.join(DATA_PATH, "cat_properties_tensor.pt"))
num_features = torch.load(os.path.join(DATA_PATH, "num_properties_tensor.pt"))

# ================= BASIC STATS =================
total_nodes = num_features.shape[0]
total_edges = edge_index.shape[1]
labeled_nodes = labels.shape[0]
total_features = cat_features.shape[1] + num_features.shape[1]

print("📊 Dataset Statistics")
print("-" * 40)
print(f"Total Nodes in Graph        : {total_nodes}")
print(f"Total Edges in Graph        : {total_edges}")
print(f"Labeled Nodes               : {labeled_nodes}")
print(f"Categorical Features        : {cat_features.shape[1]}")
print(f"Numerical Features          : {num_features.shape[1]}")
print(f"Total Features per Node     : {total_features}")
print("-" * 40)

# ================= LABEL DISTRIBUTION =================
print("\n📈 Label Distribution")
label_counts = torch.bincount(labels)
print(f"Humans (0): {label_counts[0].item()}")
print(f"Bots   (1): {label_counts[1].item()}")

# ================= SAMPLE NODE =================
print("\n🔎 Sample Node (Index 10)")
sample_index = 10

print("\nNumerical Features (Z-score normalized):")
print(num_features[sample_index])

print("\nCategorical Features:")
print(cat_features[sample_index])

# ================= GRAPH STRUCTURE =================
print("\n🔗 Graph Structure Format")
print("edge_index shape:", edge_index.shape)
print("Format: [2, num_edges]")
print("Row 0 = source nodes")
print("Row 1 = target nodes")

print("\n Dataset Demo !")