import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.explain.config import ModelConfig
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import itertools
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# RESULTS LOG
# -----------------------
# Ensure results directory exists
results_dir = "../results"
os.makedirs(results_dir, exist_ok=True)

results_log_path = os.path.join(results_dir, "results_log.csv")
if not os.path.exists(results_log_path):
    with open(results_log_path, "w") as f:
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
np.save(os.path.join(results_dir, "train_idx.npy"), train_idx)
np.save(os.path.join(results_dir, "val_idx.npy"), val_idx)
np.save(os.path.join(results_dir, "test_idx.npy"), test_idx)
print("Saved train/val/test indices to .npy files")

train_list = [data_list[i] for i in train_idx]
val_list = [data_list[i] for i in val_idx]
test_list = [data_list[i] for i in test_idx]

y_train = np.array([d.y.item() for d in train_list]).reshape(-1,1)
scaler = StandardScaler().fit(y_train)
for d in data_list:
    d.y = torch.tensor(scaler.transform(np.array([[d.y.item()]])), dtype=torch.float32).view(-1)

# -----------------------
# MODEL: GNN2
# -----------------------
class GNN2(nn.Module):
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

    model = GNN2(in_dim=in_dim, conv_type=conv_type).to(DEVICE)
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
            # Ensure models directory exists
            models_dir = "../models"
            os.makedirs(models_dir, exist_ok=True)
            
            torch.save({
                "model_state": model.state_dict(),
                "scaler": scaler,
                "in_dim": in_dim,
                "conv_type": conv_type
            }, os.path.join(models_dir, f"model_bs{bs}_lr{lr}_ep{epochs}_{conv_type}.pth"))
            pat = 0
            print("  -> Saved improved model")
        else:
            pat += 1
        if pat >= patience:
            print("Early stopping.")
            break

    with open(results_log_path, "a") as f:
        f.write(f"{bs},{lr},{epochs},{conv_type},{best_val_rmse},{test_rmse}\n")
    print(f"Logged results for BS={bs}, LR={lr}, EPOCHS={epochs}, CONV={conv_type}")


class ExplainerModelWrapper(nn.Module):
    """
    Adapts a model whose forward expects a `Data` object to the signature
    expected by the new Explainer (x, edge_index, batch=...).
    """
    def __init__(self, gnn_model):
        super().__init__()
        self.gnn = gnn_model

    def forward(self, x, edge_index, batch=None):
        # If batch is None (single graph), create zeros
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        return self.gnn(data)

# -----------------------
# EXPLAINABILITY: GNNExplainer (IMPROVED PLOTTING - FINAL FIX)
# -----------------------
# -----------------------
# EXPLAINABILITY: GNNExplainer (FINAL, ROBUST VERSION)
# -----------------------
def run_explainer_multi(model, test_list, scaler, conv_type, bs, lr, epochs,
                        explainer_epochs=200, seed=42, num_samples=5):
    """
    Generates and plots GNNExplainer explanations with a focus on clean,
    square-like, directed graphs for maximum clarity. This version uses a
    robust method for color mapping to prevent plotting errors.
    """
    model.eval()

    # --- 1. PREPARATION & DATA GATHERING ---
    print(f"Running explainer on {num_samples} random test samples...")
    sample_indices = random.sample(range(len(test_list)), num_samples)
    
    explanations, predictions = [], []
    all_node_imp, all_edge_imp = [], []
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explainer_epochs),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(mode='regression', task_level='graph', return_type='raw'),
    )

    for idx in sample_indices:
        data = test_list[idx].to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(data.x, data.edge_index, data.batch)
            pred_orig = scaler.inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).item()
            predictions.append(pred_orig)

        explanation = explainer(data.x, data.edge_index)
        explanations.append(explanation)
        all_edge_imp.extend(explanation.edge_mask.cpu().numpy())
        all_node_imp.extend(explanation.node_mask.sum(dim=1).cpu().numpy())

    # --- 2. PLOTTING SETUP ---
    fig, axes = plt.subplots(1, num_samples, figsize=(3.5 * num_samples, 4), squeeze=False)
    axes = axes.flatten()

    pos = {0: (0, 1), 1: (-0.87, -0.5), 2: (0.87, -0.5)}
    labels = {0: "A", 1: "B", 2: "C"}

    node_norm = plt.Normalize(vmin=min(all_node_imp), vmax=max(all_node_imp))
    edge_norm = plt.Normalize(vmin=min(all_edge_imp), vmax=max(all_edge_imp))
    node_cmap, edge_cmap = plt.cm.viridis, plt.cm.magma
    
    sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_edges = plt.cm.ScalarMappable(cmap=edge_cmap, norm=edge_norm)

    # --- 3. DRAW EACH SUBPLOT ---
    for i, (ax, idx, explanation, pred_val) in enumerate(zip(axes, sample_indices, explanations, predictions)):
        g = to_networkx(explanation, to_undirected=False)
        node_imp = explanation.node_mask.sum(dim=1).cpu().numpy()
        edge_imp = explanation.edge_mask.cpu().numpy()

        # ==================== THE DEFINITIVE FIX ====================
        # Manually map edge importance scores to RGBA colors before plotting.
        # This bypasses the internal library error.
        edge_rgba_colors = edge_cmap(edge_norm(edge_imp))
        # ============================================================

        # Draw nodes
        nx.draw_networkx_nodes(
            g, pos, ax=ax, node_size=800,
            node_color=node_imp, cmap=node_cmap, vmin=node_norm.vmin, vmax=node_norm.vmax,
            edgecolors='black', linewidths=1.5
        )
        
        # MODIFIED: Pass the pre-computed RGBA colors to `edge_color`.
        # REMOVED the `edge_cmap`, `edge_vmin`, and `edge_vmax` arguments.
        nx.draw_networkx_edges(
            g, pos, ax=ax,
            width=2.0,
            edge_color=edge_rgba_colors, # <-- Using the new RGBA color array
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            connectionstyle='arc3,rad=0.15'
        )
        
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=12, font_color='white', font_weight='bold')
        ax.set_title(f"Sample Index: {idx}\nPredicted Band Gap: {pred_val:.3f} eV", fontsize=10)
        
        ax.set_aspect('equal', adjustable='box')
        ax.margins(0.1)
        ax.axis('off')

    # --- 4. FINALIZE PLOT ---
    fig.suptitle("GNNExplainer: Directed Edge & Node Importance", fontsize=16, weight='bold', y=1.0)

    fig.subplots_adjust(right=0.85, wspace=0.1)
    cbar_ax_nodes = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar_ax_edges = fig.add_axes([0.92, 0.15, 0.02, 0.7])

    cbar_nodes = fig.colorbar(sm_nodes, cax=cbar_ax_nodes)
    cbar_nodes.set_label("Node Importance", fontsize=10, weight='bold')
    
    cbar_edges = fig.colorbar(sm_edges, cax=cbar_ax_edges)
    cbar_edges.set_label("Edge Importance", fontsize=10, weight='bold')
    
    # Save the plot in the GNNexplainer folder
    plot_filename = f"gnn_explanation_{conv_type}_bs{bs}_lr{lr}_ep{epochs}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Explanation plot saved as: {plot_filename}")
    
    plt.show()

print("\n--- Hyperparameter search complete. Loading best model for explanation. ---")

# 1. Read the results log
results_df = pd.read_csv(results_log_path)
results_df.columns = results_df.columns.str.strip() 

# 2. Find the single best run
best_run = results_df.loc[results_df['best_val_rmse'].idxmin()]
print(f"Best run found:\n{best_run}")


# 3. Reconstruct the model filename from the best run's parameters
bs_best = int(best_run['batch_size'])
lr_best = float(best_run['learning_rate'])
epochs_best = int(best_run['epochs'])
conv_type_best = best_run['conv_type'].strip()

best_model_path = os.path.join("../models", f"model_bs{bs_best}_lr{lr_best}_ep{epochs_best}_{conv_type_best}.pth")
print(f"Loading model from: {best_model_path}")

# 4. Load the best model's state
checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
# --- NEW: Also load the scaler from the checkpoint ---
scaler_best = checkpoint['scaler'] 
best_model = GNN2(in_dim=checkpoint['in_dim'], conv_type=checkpoint['conv_type']).to(DEVICE)
best_model.load_state_dict(checkpoint['model_state'])

# 5. Wrap the *best model* and run the explainer
wrapped_best_model = ExplainerModelWrapper(best_model)

# --- MODIFIED: Pass the loaded scaler to the function ---
run_explainer_multi(wrapped_best_model, test_list, scaler_best, conv_type_best, bs_best, lr_best, epochs_best,
                    explainer_epochs=200, seed=SEED, num_samples=5)