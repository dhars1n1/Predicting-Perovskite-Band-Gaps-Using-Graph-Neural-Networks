import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import os

MODEL_PATH = "models/model_bs16_lr0.0005_ep100_gcn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Load the model checkpoint
torch.serialization.add_safe_globals([StandardScaler])

# Load normally, allowing full checkpoint loading
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

scaler = checkpoint["scaler"]
in_dim = checkpoint["in_dim"]
conv_type = checkpoint["conv_type"]

# -----------------------
# DEFINE MODEL
# -----------------------
class GNN(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=3, dropout=0.2, conv_type="gcn"):
        super().__init__()
        self.convs = nn.ModuleList()
        ConvLayer = GCNConv if conv_type == "gcn" else SAGEConv
        self.convs.append(ConvLayer(in_dim, hidden))
        for _ in range(num_layers-1):
            self.convs.append(ConvLayer(hidden, hidden))
        self.pool = global_mean_pool
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.pool(x, batch)
        return self.mlp(x).view(-1)

model = GNN(in_dim=in_dim, conv_type=conv_type).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# -----------------------
# HELPER: Convert row to graph
# -----------------------
def row_to_graph(row, a_cols, b_cols, c_cols):
    a_vals = pd.to_numeric(row[a_cols], errors="coerce").fillna(0).astype(float).values
    b_vals = pd.to_numeric(row[b_cols], errors="coerce").fillna(0).astype(float).values
    c_vals = pd.to_numeric(row[c_cols], errors="coerce").fillna(0).astype(float).values

    a_feat, b_feat, c_feat = map(lambda v: torch.tensor(v, dtype=torch.float32), [a_vals, b_vals, c_vals])
    max_dim = max(len(a_feat), len(b_feat), len(c_feat))
    pad = lambda t: torch.cat([t, torch.zeros(max_dim - len(t))]) if len(t) < max_dim else t[:max_dim]
    a_feat, b_feat, c_feat = pad(a_feat), pad(b_feat), pad(c_feat)

    site_onehots = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)
    x = torch.stack([a_feat, b_feat, c_feat], dim=0)
    x = torch.cat([x, site_onehots], dim=1)

    edge_index = torch.tensor([
        [0,1,1,0,0,2,2,0,1,2,2,1],
        [1,0,0,1,2,0,0,2,2,0,1,1]
    ], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

# -----------------------
# LOAD NEW PEROVSKITES CSV
# -----------------------
CSV_PATH = "data/perovskite_frequency_encoded_by_site.csv"  # CSV with columns: A_*, B_*, C_*
df = pd.read_csv(CSV_PATH)
a_cols = [c for c in df.columns if c.startswith("A_")]
b_cols = [c for c in df.columns if c.startswith("B_")]
c_cols = [c for c in df.columns if c.startswith("C_")]

# -----------------------
# PREDICTIONS
# -----------------------
preds = []
graphs = []
for _, row in df.iterrows():
    graph = row_to_graph(row, a_cols, b_cols, c_cols)
    graph = graph.to(DEVICE)
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long).to(DEVICE)  # single graph batch
    with torch.no_grad():
        y_scaled = model(graph).cpu().numpy().reshape(-1,1)
        y_pred = scaler.inverse_transform(y_scaled)[0][0]
        preds.append(y_pred)
        graphs.append(graph)

df["Predicted_band_gap"] = preds

# -----------------------
# PICK BEST FOR SOLAR
# -----------------------
best_df = df.sort_values("Predicted_band_gap").reset_index(drop=True)
print("Top 5 candidate perovskites for solar cells:")
print(best_df.head(5))

# Save the graph of the best perovskite
best_graph_index = best_df.index[0]
best_graph = graphs[best_graph_index]

# Extract the elements for A, B, and C sites
best_row = best_df.iloc[0]
a_element = a_cols[np.argmax(best_row[a_cols].values)]
b_element = b_cols[np.argmax(best_row[b_cols].values)]
c_element = c_cols[np.argmax(best_row[c_cols].values)]

# -----------------------
# PLOT THE BEST PEROVSKITE GRAPH (Improved)
# -----------------------

# Convert to NetworkX
nx_graph = to_networkx(best_graph, to_undirected=True)

# Node labels
node_labels = {
    0: f"A-site\n{a_element.replace('A_', '')}",
    1: f"B-site\n{b_element.replace('B_', '')}",
    2: f"C-site\n{c_element.replace('C_', '')}"
}

# Node colors by site type
node_colors = ["#6FA8DC", "#93C47D", "#EAD1DC"]  # A: blue, B: green, C: pink

plt.figure(figsize=(7, 6))
pos = nx.spring_layout(nx_graph, seed=42, k=0.8)  # stable layout

# Draw nodes and edges
nx.draw_networkx_edges(nx_graph, pos, width=2, edge_color="gray", alpha=0.6)
nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=1500, edgecolors="black", linewidths=1.5)
nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=11, font_color="black", font_weight="bold")

# Title and aesthetics
plt.title(f"Best Perovskite Graph\nPredicted Band Gap: {best_df['Predicted_band_gap'][0]:.2f} eV", fontsize=13, pad=15)
plt.axis("off")
plt.tight_layout()
plt.savefig("results/best_perovskite_graph.png", dpi=300, bbox_inches="tight")
plt.show()

print("Graph of the best perovskite saved to results/best_perovskite_graph.png")

# -----------------------
# SAVE RESULTS
# -----------------------
df.to_csv("results/predicted_perovskites.csv", index=False)
print("Predictions saved to results/predicted_perovskites.csv")
