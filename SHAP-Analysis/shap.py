import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from torch.serialization import safe_globals

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "data/perovskite_frequency_encoded_by_site.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "plots"
MODEL_DIR = "models"
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------
# DATA LOADING
# -----------------------
df = pd.read_csv(CSV_PATH)
a_cols = [c for c in df.columns if c.startswith("A_")]
b_cols = [c for c in df.columns if c.startswith("B_")]
c_cols = [c for c in df.columns if c.startswith("C_")]

def clean_band_gap(val):
    val = str(val).replace("eV","").strip()
    if "-" in val:
        parts = val.split("-")
        try:
            nums = [float(p) for p in parts if p.strip()!=""]
            return sum(nums)/len(nums) if nums else None
        except: return None
    try: return float(val)
    except: return None

df["Perovskite_band_gap"] = df["Perovskite_band_gap"].apply(clean_band_gap)
df = df.dropna(subset=["Perovskite_band_gap"])

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
y_train = np.array([d.y.item() for d in data_list]).reshape(-1,1)
scaler = StandardScaler().fit(y_train)
for d in data_list:
    d.y = torch.tensor(scaler.transform([[d.y.item()]]), dtype=torch.float32).view(-1)
in_dim = data_list[0].x.shape[1]

# -----------------------
# MODEL DEFINITION
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

# -----------------------
# SHAP UTILS
# -----------------------
def predict_fn_wrapper(model, scaler, data_to_explain):
    model.eval()
    batch_size = data_to_explain.shape[0]
    num_nodes = 3
    num_node_features = in_dim
    x_tensor = torch.tensor(data_to_explain, dtype=torch.float32).view(batch_size, num_nodes, num_node_features).to(DEVICE)
    edge_index = torch.tensor([
        [0,1,1,0,0,2,2,0,1,2,2,1],
        [1,0,0,1,2,0,0,2,2,0,1,1]
    ], dtype=torch.long).to(DEVICE)
    graphs = [Data(x=x_tensor[i], edge_index=edge_index) for i in range(batch_size)]
    batch = Batch.from_data_list(graphs).to(DEVICE)
    with torch.no_grad():
        out = model(batch)
    return scaler.inverse_transform(out.view(-1, 1).cpu().numpy()).flatten()

def create_feature_names(a_cols, b_cols, c_cols, max_dim):
    feature_names = []
    orig_a_cols = [c.replace("A_", "") for c in a_cols]
    orig_b_cols = [c.replace("B_", "") for c in b_cols]
    orig_c_cols = [c.replace("C_", "") for c in c_cols]
    for col in orig_a_cols:
        feature_names.append(f"A-site_{col}")
    for i in range(len(orig_a_cols), max_dim):
        feature_names.append(f"A-site_pad_{i}")
    feature_names.extend(["A-site_is_A", "A-site_is_B", "A-site_is_C"])
    for col in orig_b_cols:
        feature_names.append(f"B-site_{col}")
    for i in range(len(orig_b_cols), max_dim):
        feature_names.append(f"B-site_pad_{i}")
    feature_names.extend(["B-site_is_A", "B-site_is_B", "B-site_is_C"])
    for col in orig_c_cols:
        feature_names.append(f"C-site_{col}")
    for i in range(len(orig_c_cols), max_dim):
        feature_names.append(f"C-site_pad_{i}")
    feature_names.extend(["C-site_is_A", "C-site_is_B", "C-site_is_C"])
    return feature_names

# -----------------------
# MAIN LOGIC
# -----------------------
def run_shap_analysis():
    results_csv = os.path.join(PLOT_DIR, "model_evaluation_results.csv")
    df_res = pd.read_csv(results_csv)
    print("Columns in model_evaluation_results.csv:", df_res.columns.tolist())
    best_model_row = df_res.loc[df_res['test_rmse'].idxmin()]
    best_model_path = os.path.join(MODEL_DIR, best_model_row["model"])
    print(f"\nBest Model: {best_model_row['model']} | Test RMSE: {best_model_row['test_rmse']:.4f}")
    # Load best model with safe_globals and weights_only=False
    with safe_globals([StandardScaler]):
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model_to_explain = GNN2(in_dim=in_dim, conv_type=best_model_row["conv_type"]).to(DEVICE)
    model_to_explain.load_state_dict(checkpoint["model_state"])
    # Prepare SHAP data
    all_node_features = np.stack([d.x.flatten().cpu().numpy() for d in data_list])
    max_dim = max(len(a_cols), len(b_cols), len(c_cols))
    feature_names = create_feature_names(a_cols, b_cols, c_cols, max_dim)
    np.random.seed(42)
    background_data = all_node_features[np.random.choice(all_node_features.shape[0], 50, replace=False)]
    n_samples = min(100, len(data_list))
    sample_indices = np.random.choice(len(data_list), n_samples, replace=False)
    samples_to_explain = all_node_features[sample_indices]
    print(f"\nRunning SHAP analysis on {n_samples} samples...")
    explainer = shap.KernelExplainer(lambda x: predict_fn_wrapper(model_to_explain, scaler, x), background_data)
    shap_values = explainer.shap_values(samples_to_explain, nsamples=100)
    # Save SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, samples_to_explain, feature_names=feature_names, show=False, max_display=25)
    plt.title("SHAP Feature Importance - Perovskite Band Gap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary_plot0610.png"), dpi=300)
    plt.close()
    print(f"✓ Saved SHAP summary plot to {PLOT_DIR}/shap_summary_plot0610.png")
    # Save feature importance ranking with all requested columns
    mean_shap = np.mean(shap_values, axis=0)
    std_shap = np.std(shap_values, axis=0)
    abs_mean_shap = np.mean(np.abs(shap_values), axis=0)
    importance_score = abs_mean_shap
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_score': importance_score,
        'mean_shap_value': mean_shap,
        'std_shap_value': std_shap,
        'abs_mean_shap': abs_mean_shap
    }).sort_values(by='importance_score', ascending=False)
    feature_importance_df.to_csv(os.path.join(PLOT_DIR, "feature_importance_ranking0610.csv"), index=False)
    print(f"✓ Saved feature importance to {PLOT_DIR}/feature_importance_ranking0610.csv")
    # SHAP correlation analysis (top N features only)
    TOP_N = 10
    top_features = feature_importance_df['feature_name'].head(TOP_N).tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    shap_values_top = shap_values[:, top_indices]
    shap_corr = np.corrcoef(shap_values_top.T)
    plt.figure(figsize=(8, 6))
    plt.imshow(shap_corr, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title(f"SHAP Feature Correlation (Top {TOP_N})")
    plt.xticks(range(TOP_N), top_features, rotation=90, fontsize=8)
    plt.yticks(range(TOP_N), top_features, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_correlation_top.png"), dpi=300)
    plt.close()
    print(f"✓ Saved SHAP correlation plot (top features) to {PLOT_DIR}/shap_correlation_top.png")
    # SHAP site analysis (top N features only)
    plt.figure(figsize=(6, 4))
    site_means = {}
    for site in ['A', 'B', 'C']:
        site_feats = [f for f in top_features if f.startswith(f"{site}-site") or f.startswith(f"{site}_site")]
        idxs = [feature_names.index(f) for f in site_feats if f in feature_names]
        if idxs:
            site_means[site] = np.mean(abs_mean_shap[idxs])
        else:
            site_means[site] = 0
    plt.bar(site_means.keys(), site_means.values())
    plt.ylabel("Mean |SHAP value|")
    plt.title(f"Mean Absolute SHAP Value by Site (Top {TOP_N} Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_site_analysis_top.png"), dpi=300)
    plt.close()
    print(f"✓ Saved SHAP site analysis plot (top features) to {PLOT_DIR}/shap_site_analysis_top.png")
    print("\n🎉 SHAP Analysis Complete!")

if __name__ == "__main__":
    run_shap_analysis()
