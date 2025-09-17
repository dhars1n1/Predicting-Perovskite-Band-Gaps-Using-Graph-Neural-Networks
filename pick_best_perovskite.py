import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.serialization import safe_globals


import torch
MODEL_PATH = "models/model_bs16_lr0.0005_ep100_gcn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Force full loading of the checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

# # Allowlist all necessary globals from previous errors
# with safe_globals([StandardScaler, np._core.multiarray.scalar, np.dtype]):
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

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
for _, row in df.iterrows():
    graph = row_to_graph(row, a_cols, b_cols, c_cols)
    graph = graph.to(DEVICE)
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long).to(DEVICE)  # single graph batch
    with torch.no_grad():
        y_scaled = model(graph).cpu().numpy().reshape(-1,1)
        y_pred = scaler.inverse_transform(y_scaled)[0][0]
        preds.append(y_pred)

df["Predicted_band_gap"] = preds

# -----------------------
# PICK BEST FOR SOLAR
# -----------------------
best_df = df.sort_values("Predicted_band_gap").reset_index(drop=True)
print("Top 5 candidate perovskites for solar cells:")
print(best_df.head(5))

# -----------------------
# SAVE RESULTS
# -----------------------
df.to_csv("results/predicted_perovskites.csv", index=False)
print("Predictions saved to results/predicted_perovskites.csv")
