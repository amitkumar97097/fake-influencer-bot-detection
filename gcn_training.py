import json
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🚀 Loading TwiBot Dataset...")

with open("TwiBot-20_sample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Users Loaded:", len(data))

# ==============================
# STEP 1 — Extract Features
# ==============================

records = []
edges = []

for user in data:
    profile = user.get("profile", {})
    neighbor = user.get("neighbor", {})
    domain = user.get("domain", [])

    user_id = str(user.get("ID"))

    record = {
        "user_id": user_id,
        "followers": int(profile.get("followers_count", 0)),
        "friends": int(profile.get("friends_count", 0)),
        "statuses": int(profile.get("statuses_count", 0)),
        "verified": 1 if profile.get("verified", "False").strip() == "True" else 0,
        "label": 1 if "Bot" in str(domain) else 0
    }

    records.append(record)

    # Build edges from following relationships
    if neighbor and neighbor.get("following"):
        for target in neighbor["following"]:
            edges.append((user_id, str(target)))

df = pd.DataFrame(records)

print("Feature Extraction Done")
print("Label Distribution:")
print(df["label"].value_counts())

# ==============================
# STEP 2 — Build Graph
# ==============================

G = nx.Graph()

for _, row in df.iterrows():
    G.add_node(row["user_id"])

for edge in edges:
    if edge[1] in G.nodes:
        G.add_edge(edge[0], edge[1])

print("Graph Created")
print("Total Nodes:", G.number_of_nodes())
print("Total Edges:", G.number_of_edges())

# ==============================
# STEP 3 — Prepare PyG Data
# ==============================

node_mapping = {node: i for i, node in enumerate(G.nodes)}
reverse_mapping = {i: node for node, i in node_mapping.items()}

edge_index = []

for source, target in G.edges():
    edge_index.append([node_mapping[source], node_mapping[target]])
    edge_index.append([node_mapping[target], node_mapping[source]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

features = df.set_index("user_id").loc[list(G.nodes)]
X = features[["followers", "friends", "statuses", "verified"]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(features["label"].values, dtype=torch.long)

data = Data(x=X, edge_index=edge_index, y=y)

# ==============================
# STEP 4 — Train/Test Split
# ==============================

train_idx, test_idx = train_test_split(
    range(len(y)), test_size=0.2, random_state=42
)

train_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask = torch.zeros(len(y), dtype=torch.bool)

train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

# ==============================
# STEP 5 — Define GCN Model
# ==============================

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

model = GCN(num_features=data.num_features, hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# ==============================
# STEP 6 — Training Loop
# ==============================

print("🔥 Training GCN Model...")

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==============================
# STEP 7 — Evaluation
# ==============================

model.eval()
pred = model(data).argmax(dim=1)

correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())

print("\n📈 GCN Test Accuracy:", acc)
print("🎉 GCN Training Completed Successfully!")