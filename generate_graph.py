import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import os

print("Loading dataset...")
df = pd.read_csv("Twitter_DataSET/bot_detection_data.csv")

# Only keep Username and Tweet
df_small = df[["Username", "Tweet"]]

G = nx.DiGraph()

print("Building graph...")

for _, row in df_small.iterrows():
    user = row["Username"]
    tweet = str(row["Tweet"])

    # Extract mentions using @username pattern
    mentions = re.findall(r'@([A-Za-z0-9_]+)', tweet)

    # Add user node
    G.add_node(user)

    # Add edges: user -> mentioned user
    for m in mentions:
        G.add_edge(user, m)

print("Graph created!")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# Keep a small subgraph to visualize
sub_nodes = list(G.nodes())[:50]   # visualize only first 50
H = G.subgraph(sub_nodes)

# Create output folder
os.makedirs("graph_outputs", exist_ok=True)

plt.figure(figsize=(12, 10))
nx.draw(H, with_labels=True, node_size=500, node_color="skyblue", font_size=6)
plt.title("Twitter Mention Graph (Sample)")
plt.savefig("graph_outputs/mention_graph.png", dpi=300)
plt.close()

print("Graph image saved at: graph_outputs/mention_graph.png")
