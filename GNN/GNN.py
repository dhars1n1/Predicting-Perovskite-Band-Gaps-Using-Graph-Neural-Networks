import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
import itertools
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# RESULTS LOG
# -----------------------
results_dir = "../results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(f"{results_dir}/results_log.csv"):
    with open(f"{results_dir}/results_log.csv", "w") as f:
        f.write("batch_size,learning_rate,epochs,conv_type,best_val_rmse,test_rmse\n")

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "../data/perovskite_frequency_encoded_by_site.csv"
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 15

torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(CSV_PATH)
if "Perovskite_band_gap" not in df.columns:
    raise ValueError("CSV must contain 'Perovskite_band_gap' column")

a_cols = [c for c in df.columns if c.startswith("A_")]
b_cols = [c for c in df.columns if c.startswith("B_")]
c_cols = [c for c in df.columns if c.startswith("C_")]

if not (a_cols and b_cols and c_cols):
    raise ValueError("Could not find columns with prefixes A_, B_, C_ in CSV.")

print(f"Found {len(a_cols)} A-site, {len(b_cols)} B-site, {len(c_cols)} C-site features.")

def clean_band_gap(val):
    val = str(val).replace("eV", "").strip()
    if "-" in val:
        parts = val.split("-")
        try:
            nums = [float(p) for p in parts if p.strip() != ""]
            return sum(nums) / len(nums) if nums else None
        except:
            return None
    try:
        return float(val)
    except:
        return None

df["Perovskite_band_gap"] = df["Perovskite_band_gap"].apply(clean_band_gap)
df = df.dropna(subset=["Perovskite_band_gap"])

# -----------------------
# GRAPH CONSTRUCTION
# -----------------------
def row_to_graph(row):
    a_vals = pd.to_numeric(row[a_cols], errors="coerce").fillna(0).astype(float).values
    b_vals = pd.to_numeric(row[b_cols], errors="coerce").fillna(0).astype(float).values
    c_vals = pd.to_numeric(row[c_cols], errors="coerce").fillna(0).astype(float).values

    a_feat, b_feat, c_feat = map(lambda v: torch.tensor(v, dtype=torch.float32), [a_vals, b_vals, c_vals])

    max_dim = max(len(a_feat), len(b_feat), len(c_feat))
    def pad_tensor(t, size):
        return torch.cat([t, torch.zeros(size - len(t))]) if len(t) < size else t[:size]

    a_feat, b_feat, c_feat = pad_tensor(a_feat, max_dim), pad_tensor(b_feat, max_dim), pad_tensor(c_feat, max_dim)

    site_onehots = torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)
    x = torch.stack([a_feat, b_feat, c_feat], dim=0)
    x = torch.cat([x, site_onehots], dim=1)

    edge_index = torch.tensor([
        [0,1,1,0,0,2,2,0,1,2,2,1],
        [1,0,0,1,2,0,0,2,2,0,1,1]
    ], dtype=torch.long)

    y = torch.tensor([row["Perovskite_band_gap"]], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, y=y)

data_list = [row_to_graph(row) for _, row in df.iterrows()]
print(f"Built {len(data_list)} graphs.")

# -----------------------
# SPLIT DATA
# -----------------------
train_val_idx, test_idx = train_test_split(np.arange(len(data_list)), test_size=TEST_SIZE, random_state=SEED)
train_idx, val_idx = train_test_split(train_val_idx, test_size=VAL_SIZE/(1 - TEST_SIZE), random_state=SEED)

# Save indices for reproducibility
np.save(f"{results_dir}/train_idx.npy", train_idx)
np.save(f"{results_dir}/val_idx.npy", val_idx)
np.save(f"{results_dir}/test_idx.npy", test_idx)
print("Saved train/val/test indices to .npy files")

train_list = [data_list[i] for i in train_idx]
val_list = [data_list[i] for i in val_idx]
test_list = [data_list[i] for i in test_idx]

y_train = np.array([d.y.item() for d in train_list]).reshape(-1,1)
scaler = StandardScaler().fit(y_train)
for d in data_list:
    d.y = torch.tensor(scaler.transform(np.array([[d.y.item()]])), dtype=torch.float32).view(-1)

# -----------------------
# MODEL: GNN
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

in_dim = data_list[0].x.shape[1]

# -----------------------
# EVALUATION
# -----------------------
def evaluate(loader, model, scaler):
    model.eval()
    losses, preds, targets = [], [], []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            losses.append(loss.item() * batch.num_graphs)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    if not losses:
        return None, None
    preds = np.concatenate(preds).reshape(-1,1)
    targets = np.concatenate(targets).reshape(-1,1)
    preds_orig, targets_orig = scaler.inverse_transform(preds), scaler.inverse_transform(targets)
    rmse = np.sqrt(np.mean((preds_orig - targets_orig)**2))
    return np.mean(losses) / len(loader.dataset), rmse

# -----------------------
# HYPERPARAMETER SEARCH
# -----------------------
hyperparams_grid = {
    "BATCH_SIZE": [16, 32],
    "LR": [1e-3, 5e-4],
    "EPOCHS": [50, 100],
    "CONV_TYPE": ["gcn", "sage"]
}

for bs, lr, epochs, conv_type in itertools.product(
        hyperparams_grid["BATCH_SIZE"],
        hyperparams_grid["LR"],
        hyperparams_grid["EPOCHS"],
        hyperparams_grid["CONV_TYPE"]):
    print(f"\n--- Running with BS={bs}, LR={lr}, EPOCHS={epochs}, CONV={conv_type} ---")
    train_loader = DataLoader(train_list, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_list, batch_size=bs)
    test_loader  = DataLoader(test_list, batch_size=bs)

    model = GNN(in_dim=in_dim, conv_type=conv_type).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse, pat = 1e9, 0
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_rmse = evaluate(val_loader, model, scaler)
        test_loss, test_rmse = evaluate(test_loader, model, scaler)
        print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val RMSE {val_rmse:.4f} | Test RMSE {test_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
                "in_dim": in_dim,
                "conv_type": conv_type
            }, f"../models/model_bs{bs}_lr{lr}_ep{epochs}_{conv_type}.pth")
            pat = 0
            print("  -> Saved improved model")
        else:
            pat += 1
        if pat >= patience:
            print("Early stopping.")
            break

    with open(f"{results_dir}/results_log.csv", "a") as f:
        f.write(f"{bs},{lr},{epochs},{conv_type},{best_val_rmse},{test_rmse}\n")
    print(f"Logged results for BS={bs}, LR={lr}, EPOCHS={epochs}, CONV={conv_type}")
