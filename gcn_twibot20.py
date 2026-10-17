# ============================================================
# Fake Influencer Detection using GraphSAGE & Hybrid Models
# TwiBot-20 Dataset
# Reproducible + Statistical Research Implementation (FINAL)
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ===========================
# 1️⃣ Reproducibility
# ===========================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===========================
# 2️⃣ Load Dataset
# ===========================
print("🚀 Loading TwiBot-20 Dataset...")

edge_index = torch.load("processed_data/edge_index.pt")
labels = torch.load("processed_data/label.pt")
cat_features = torch.load("processed_data/cat_properties_tensor.pt")
num_features = torch.load("processed_data/num_properties_tensor.pt")

x = torch.cat([cat_features, num_features], dim=1)
data = Data(x=x, edge_index=edge_index, y=labels).to(device)

num_labeled = labels.shape[0]

print("Total Nodes:", x.shape[0])
print("Total Edges:", edge_index.shape[1])
print("Labeled Nodes:", num_labeled)


# ===========================
# 3️⃣ Train/Test Split
# ===========================
train_size = int(0.8 * num_labeled)
perm = torch.randperm(num_labeled)
train_idx = perm[:train_size]
test_idx = perm[train_size:]


# ===========================
# 4️⃣ Class Weights
# ===========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels.numpy()),
    y=labels.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


# ===========================
# 5️⃣ Model Definitions
# ===========================
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.dropout(F.relu(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.dropout(F.relu(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)
        return x


# ============================================================
# 🔬 ABLATION STUDY
# ============================================================
hidden_dims = [64, 128, 256]
print("\n🔬 Running Hidden Dimension Ablation")

for hidden in hidden_dims:
    model = GraphSAGE(x.shape[1], hidden, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(dim=1)

    acc = accuracy_score(data.y[test_idx].cpu(), preds[test_idx].cpu())
    fpr, tpr, _ = roc_curve(data.y[test_idx].cpu(), probs[test_idx][:, 1].cpu())
    auc_score = auc(fpr, tpr)

    print(f"Hidden={hidden} | Accuracy={acc:.4f} | AUC={auc_score:.4f}")


# ============================================================
# 🔁 5-SEED STATISTICAL VALIDATION
# ============================================================
print("\n🔁 Running 5-Seed Statistical Evaluation (hidden=256)")

def run_graphsage(seed, hidden_dim=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    perm = torch.randperm(num_labeled)
    train_idx_local = perm[:train_size]
    test_idx_local = perm[train_size:]

    model = GraphSAGE(x.shape[1], hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx_local], data.y[train_idx_local], weight=class_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(dim=1)

    acc = accuracy_score(data.y[test_idx_local].cpu(), preds[test_idx_local].cpu())
    fpr, tpr, _ = roc_curve(data.y[test_idx_local].cpu(), probs[test_idx_local][:, 1].cpu())
    auc_score = auc(fpr, tpr)

    return acc, auc_score


seeds = [0,1,2,3,4]
acc_list = []
auc_list = []

for s in seeds:
    acc, auc_score = run_graphsage(s)
    acc_list.append(acc)
    auc_list.append(auc_score)
    print(f"Seed {s} → Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

print("\n📊 Final Statistical Results")
print(f"Mean Accuracy: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
print(f"Mean AUC: {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")


# ============================================================
# 🔁 Train Final GraphSAGE (for ROC + Hybrid)
# ============================================================
print("\n🔁 Training Final GraphSAGE (hidden=256)")
#torch.save(final_model.state_dict(), "ml_outputs/graphsage_model.pt")
#print("GraphSAGE model saved successfully!")

final_model = GraphSAGE(x.shape[1], 256, 2).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(200):
    final_model.train()
    optimizer.zero_grad()
    out = final_model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx], weight=class_weights)
    loss.backward()
    optimizer.step()

final_model.eval()

with torch.no_grad():
    final_out = final_model(data.x, data.edge_index)
    final_probs = torch.softmax(final_out, dim=1)
    
# Save trained GraphSAGE model
torch.save(final_model.state_dict(), "ml_outputs/graphsage_model.pt")

print("✅ GraphSAGE model saved successfully!")


# ============================================================
# 📘 GCN BASELINE
# ============================================================
print("\n🔥 Training GCN")

gcn_model = GCN(x.shape[1], 128, 2).to(device)
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(200):
    gcn_model.train()
    optimizer.zero_grad()
    out = gcn_model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx], weight=class_weights)
    loss.backward()
    optimizer.step()

gcn_model.eval()
with torch.no_grad():
    gcn_out = gcn_model(data.x, data.edge_index)
    gcn_probs = torch.softmax(gcn_out, dim=1)

gcn_acc = accuracy_score(data.y[test_idx].cpu(), gcn_probs.argmax(dim=1)[test_idx].cpu())
print("GCN Accuracy:", round(gcn_acc, 4))


# ============================================================
# 📊 Feature Baseline
# ============================================================
X = x[:num_labeled].numpy()
y = labels.numpy()

X_train = X[train_idx.numpy()]
X_test = X[test_idx.numpy()]
y_train = y[train_idx.numpy()]
y_test = y[test_idx.numpy()]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:,1]


# ============================================================
# 🔗 Improved Hybrid Representation Learning Model
# ============================================================

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("\n🚀 Training Improved Hybrid Model (GraphSAGE + Features + MLP)")

# ------------------------------------------------
# Extract GraphSAGE embeddings
# ------------------------------------------------
with torch.no_grad():

    h1 = F.relu(final_model.conv1(data.x, data.edge_index))
    h2 = F.relu(final_model.conv2(h1, data.edge_index))

graph_embeddings = h2.cpu().numpy()

print("Graph Embedding Shape:", graph_embeddings.shape)

# ------------------------------------------------
# Combine Original Features + Graph Embeddings
# ------------------------------------------------
X_comb = np.concatenate(
    [X, graph_embeddings[:num_labeled]],
    axis=1
)

print("Combined Feature Shape:", X_comb.shape)

# ------------------------------------------------
# Train/Test Split
# ------------------------------------------------
X_train_comb = X_comb[train_idx.numpy()]
X_test_comb = X_comb[test_idx.numpy()]

# ------------------------------------------------
# Feature Scaling
# ------------------------------------------------
hybrid_scaler = StandardScaler()

X_train_comb = hybrid_scaler.fit_transform(X_train_comb)
X_test_comb = hybrid_scaler.transform(X_test_comb)

# ------------------------------------------------
# MLP Hybrid Classifier
# ------------------------------------------------
hybrid_mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True
)

# ------------------------------------------------
# Train Model
# ------------------------------------------------
hybrid_mlp.fit(X_train_comb, y_train)

print("✅ Hybrid MLP Training Completed")

# ------------------------------------------------
# Predictions
# ------------------------------------------------
hybrid_preds = hybrid_mlp.predict(X_test_comb)

hybrid_probs = hybrid_mlp.predict_proba(X_test_comb)[:,1]

# ------------------------------------------------
# Evaluation
# ------------------------------------------------
hybrid_acc = accuracy_score(y_test, hybrid_preds)

print("\n📈 Hybrid Model Accuracy:", round(hybrid_acc, 4))

print("\n📋 Classification Report:\n")
print(classification_report(y_test, hybrid_preds))

print("\n🧩 Confusion Matrix:\n")
print(confusion_matrix(y_test, hybrid_preds))
# ============================================================
# 📈 ROC Comparison
# ============================================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_gs, tpr_gs, _ = roc_curve(y_test, final_probs[test_idx][:,1].cpu())
fpr_hb, tpr_hb, _ = roc_curve(y_test, hybrid_probs)

auc_lr = auc(fpr_lr, tpr_lr)
auc_gs = auc(fpr_gs, tpr_gs)
auc_hb = auc(fpr_hb, tpr_hb)

print("\nAUC Logistic Regression:", round(auc_lr,4))
print("AUC GraphSAGE:", round(auc_gs,4))
print("AUC Hybrid:", round(auc_hb,4))

plt.figure(figsize=(7,6))
plt.plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={auc_lr:.3f})")
plt.plot(fpr_gs, tpr_gs, label=f"GraphSAGE (AUC={auc_gs:.3f})")
plt.plot(fpr_hb, tpr_hb, label=f"Hybrid (AUC={auc_hb:.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

print("\n🎉 COMPLETE RESEARCH PIPELINE EXECUTED SUCCESSFULLY!")