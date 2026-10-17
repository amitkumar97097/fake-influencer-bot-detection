import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import os

print("Loading dataset...")
df = pd.read_csv("Twitter_DataSET/bot_detection_data.csv")

# Select first 300 users (easy to visualize)
users = df["Username"].unique()[:300]

G = nx.DiGraph()
G.add_nodes_from(users)

print("Simulating mention edges...")

# Randomly add edges to simulate interactions
for i in range(len(users)):
    user = users[i]

    # each user mentions 1–3 random users
    for _ in range(random.randint(1, 3)):
        target = random.choice(users)
        if user != target:
            G.add_edge(user, target)

print("Graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

os.makedirs("graph_outputs", exist_ok=True)

plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=False,
    node_color="skyblue",
    node_size=120,
    edge_color="gray",
    alpha=0.7
)

plt.title("Simulated Social Graph of Users")
plt.savefig("graph_outputs/enhanced_graph.png", dpi=300)
plt.close()

print("Enhanced graph saved at: graph_outputs/enhanced_graph.png")
