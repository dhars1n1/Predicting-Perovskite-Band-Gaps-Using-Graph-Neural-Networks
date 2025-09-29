import os
import re
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.serialization import safe_globals
import shap

# -----------------------
# CONFIG
# -----------------------
CSV_PATH = "data/perovskite_frequency_encoded_by_site.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
a_cols = [c for c in df.columns if c.startswith("A_")]
b_cols = [c for c in df.columns if c.startswith("B_")]
c_cols = [c for c in df.columns if c.startswith("C_")]

print("Available columns in dataset:")
print(f"A-site features ({len(a_cols)}): {a_cols[:5]}{'...' if len(a_cols) > 5 else ''}")
print(f"B-site features ({len(b_cols)}): {b_cols[:5]}{'...' if len(b_cols) > 5 else ''}")
print(f"C-site features ({len(c_cols)}): {c_cols[:5]}{'...' if len(c_cols) > 5 else ''}")

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
print(f"Built {len(data_list)} graphs.")

y_train = np.array([d.y.item() for d in data_list]).reshape(-1,1)
scaler = StandardScaler().fit(y_train)
for d in data_list:
    d.y = torch.tensor(scaler.transform([[d.y.item()]]), dtype=torch.float32).view(-1)

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
        # Handle both Data objects and raw tensors for SHAP compatibility
        if isinstance(data, torch.Tensor):
            x = data
            # For SHAP, use fixed graph structure
            edge_index = torch.tensor([
                [0,1,1,0,0,2,2,0,1,2,2,1],
                [1,0,0,1,2,0,0,2,2,0,1,1]
            ], dtype=torch.long, device=x.device)
            batch = torch.tensor([0, 0, 0], dtype=torch.long, device=x.device)
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.pool(x, batch)
        return self.mlp(x).view(-1)

in_dim = data_list[0].x.shape[1]
test_loader = DataLoader(data_list, batch_size=32)

def evaluate(loader, model, scaler):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds).reshape(-1,1)
    targets = np.concatenate(targets).reshape(-1,1)
    preds_orig = scaler.inverse_transform(preds)
    targets_orig = scaler.inverse_transform(targets)
    rmse = np.sqrt(np.mean((preds_orig - targets_orig)**2))
    return rmse

def predict_fn_wrapper(model, scaler, data_to_explain):
    """
    Wrapper function for SHAP explainer
    """
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
    batch = torch_geometric.data.Batch.from_data_list(graphs).to(DEVICE)

    with torch.no_grad():
        out = model(batch)

    return scaler.inverse_transform(out.view(-1, 1).cpu().numpy()).flatten()

def create_feature_names(a_cols, b_cols, c_cols, max_dim):
    """
    Create meaningful feature names based on your actual dataset
    """
    feature_names = []
    orig_a_cols = [c.replace("A_", "") for c in a_cols]
    orig_b_cols = [c.replace("B_", "") for c in b_cols]
    orig_c_cols = [c.replace("C_", "") for c in c_cols]

    # A-site features
    for col in orig_a_cols:
        feature_names.append(f"A-site_{col}")
    # A-site padding
    for i in range(len(orig_a_cols), max_dim):
        feature_names.append(f"A-site_pad_{i}")
    # A-site one-hot encoding
    feature_names.extend(["A-site_is_A", "A-site_is_B", "A-site_is_C"])

    # B-site features
    for col in orig_b_cols:
        feature_names.append(f"B-site_{col}")
    # B-site padding
    for i in range(len(orig_b_cols), max_dim):
        feature_names.append(f"B-site_pad_{i}")
    # B-site one-hot encoding
    feature_names.extend(["B-site_is_A", "B-site_is_B", "B-site_is_C"])

    # C-site features
    for col in orig_c_cols:
        feature_names.append(f"C-site_{col}")
    # C-site padding
    for i in range(len(orig_c_cols), max_dim):
        feature_names.append(f"C-site_pad_{i}")
    # C-site one-hot encoding
    feature_names.extend(["C-site_is_A", "C-site_is_B", "C-site_is_C"])

    return feature_names

def run_comprehensive_shap_analysis():
    """
    Run comprehensive SHAP analysis on your perovskite models
    """

    model_files = glob.glob("models/model2_*.pth")
    if not model_files:
        print("No model files found! Make sure your trained models are in 'models/' directory")
        return

    results = []

    print("Evaluating all models...")
    for fpath in model_files:
        m = re.match(r".*model2_bs(\d+)_lr([\d.]+)_ep(\d+)_(\w+)\.pth", fpath)
        if not m: continue
        bs, lr, ep, conv_type = int(m[1]), float(m[2]), int(m[3]), m[4]

        with safe_globals([StandardScaler]):
            checkpoint = torch.load(fpath, map_location=DEVICE, weights_only=False)

        model = GNN2(in_dim=in_dim, conv_type=conv_type).to(DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        rmse = evaluate(test_loader, model, scaler)
        results.append({
            "model": os.path.basename(fpath),
            "conv_type": conv_type,
            "bs": bs, "lr": lr, "epochs": ep,
            "test_rmse": rmse
        })
        print(f"{os.path.basename(fpath)} | Test RMSE: {rmse:.4f}")

    df_res = pd.DataFrame(results)
    best_model_row = df_res.loc[df_res['test_rmse'].idxmin()]
    best_model_path = best_model_row["model"]

    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_path}")
    print(f"Test RMSE: {best_model_row['test_rmse']:.4f}")
    print(f"{'='*50}")

    with safe_globals([StandardScaler]):
        checkpoint = torch.load(f"models/{best_model_path}", map_location=DEVICE, weights_only=False)

    model_to_explain = GNN2(in_dim=in_dim, conv_type=best_model_row["conv_type"]).to(DEVICE)
    model_to_explain.load_state_dict(checkpoint["model_state"])

    all_node_features = []
    for data in data_list:
        all_node_features.append(data.x.flatten().cpu().numpy())
    all_node_features = np.stack(all_node_features)

    max_dim = max(len(a_cols), len(b_cols), len(c_cols))
    feature_names = create_feature_names(a_cols, b_cols, c_cols, max_dim)

    print(f"\nFeature Analysis:")
    print(f"Total features: {len(feature_names)}")
    print(f"Feature vector size: {all_node_features.shape[1]}")
    print("Sample feature names:", feature_names[:10])

    np.random.seed(42)
    background_data = all_node_features[np.random.choice(all_node_features.shape[0], 50, replace=False)]

    n_samples = min(100, len(data_list))
    sample_indices = np.random.choice(len(data_list), n_samples, replace=False)
    samples_to_explain = all_node_features[sample_indices]

    print(f"\nRunning SHAP analysis on {n_samples} samples...")

    explainer = shap.KernelExplainer(
        lambda x: predict_fn_wrapper(model_to_explain, scaler, x),
        background_data
    )

    shap_values = explainer.shap_values(samples_to_explain, nsamples=100)
    predictions = predict_fn_wrapper(model_to_explain, scaler, samples_to_explain)
    true_values = []
    for idx in sample_indices:
        true_val = scaler.inverse_transform(data_list[idx].y.view(-1, 1).cpu().numpy())[0, 0]
        true_values.append(true_val)
    true_values = np.array(true_values)

    print(f"Analysis Results:")
    print(f"Mean prediction: {np.mean(predictions):.4f} eV")
    print(f"Mean true value: {np.mean(true_values):.4f} eV")
    print(f"Prediction RMSE: {np.sqrt(np.mean((predictions - true_values)**2)):.4f} eV")

    feature_importance = np.mean(np.abs(shap_values), axis=0)

    top_indices = np.argsort(feature_importance)[-25:]
    top_features = [(i, feature_names[i], feature_importance[i]) for i in top_indices]

    print(f"\nTop 15 Most Important Features:")
    for i, (idx, name, importance) in enumerate(sorted(top_features, key=lambda x: x[2], reverse=True)[:15]):
        print(f"{i+1:2d}. {name:<30} | Importance: {importance:.6f}")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values[:, top_indices],
        samples_to_explain[:, top_indices],
        feature_names=[feature_names[i] for i in top_indices],
        show=False,
        max_display=25
    )
    plt.title("SHAP Analysis - Perovskite Band Gap Prediction", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved SHAP summary plot to {PLOT_DIR}/shap_summary_plot.png")

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    site_groups = [
        ([i for i, name in enumerate(feature_names) if name.startswith('A-site_') and not 'pad' in name and not 'is_' in name], "A-site Features", axes[0]),
        ([i for i, name in enumerate(feature_names) if name.startswith('B-site_') and not 'pad' in name and not 'is_' in name], "B-site Features", axes[1]),
        ([i for i, name in enumerate(feature_names) if name.startswith('C-site_') and not 'pad' in name and not 'is_' in name], "C-site Features", axes[2])
    ]

    for site_indices, title, ax in site_groups:
        if len(site_indices) > 0:
            site_importance = feature_importance[site_indices]
            top_site_indices = np.array(site_indices)[np.argsort(site_importance)[-10:]]

            for i, idx in enumerate(top_site_indices):
                y_pos = len(top_site_indices) - i - 1
                shap_vals = shap_values[:, idx]
                feature_vals = samples_to_explain[:, idx]

                scatter = ax.scatter(shap_vals, [y_pos] * len(shap_vals),
                                   c=feature_vals, cmap='RdYlBu_r', alpha=0.6, s=20)

            ax.set_yticks(range(len(top_site_indices)))
            ax.set_yticklabels([feature_names[i].replace(f"{title.split()[0]}_", "") for i in top_site_indices])
            ax.set_xlabel('SHAP value (impact on model output)')
            ax.set_title(title, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_site_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved site-based analysis to {PLOT_DIR}/shap_site_analysis.png")

    top_2_features = np.argsort(feature_importance)[-2:]
    feat1_idx, feat2_idx = top_2_features[1], top_2_features[0]

    feat1_name = feature_names[feat1_idx]
    feat2_name = feature_names[feat2_idx]
    feat1_values = samples_to_explain[:, feat1_idx]
    feat1_shap = shap_values[:, feat1_idx]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        feat1_values,
        feat1_shap,
        c=predictions,
        cmap='viridis',
        alpha=0.7,
        s=50,
        edgecolors='black',
        linewidth=0.3
    )

    plt.colorbar(scatter, label='Predicted Band Gap (eV)')

    feat1_clean = feat1_name.replace('A-site_', '').replace('B-site_', '').replace('C-site_', '')

    plt.xlabel(f'{feat1_clean} (Feature Value)')
    plt.ylabel(f'SHAP Impact for {feat1_clean}')
    plt.title(f'Feature-SHAP Correlation Analysis\nMost Important Feature: {feat1_clean}')
    plt.grid(True, alpha=0.3)

    try:
        z = np.polyfit(feat1_values, feat1_shap, 1)
        p = np.poly1d(z)
        plt.plot(sorted(feat1_values), p(sorted(feat1_values)), "r--", alpha=0.8, linewidth=2, label='Trend')
        plt.legend()
    except:
        pass

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "shap_correlation_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved correlation analysis to {PLOT_DIR}/shap_correlation_analysis.png")

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df['prediction'] = predictions
    shap_df['true_value'] = true_values
    shap_df['sample_index'] = sample_indices
    shap_df.to_csv(os.path.join(PLOT_DIR, "shap_detailed_results.csv"), index=False)

    importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance_score': feature_importance,
        'mean_shap_value': np.mean(shap_values, axis=0),
        'std_shap_value': np.std(shap_values, axis=0),
        'abs_mean_shap': np.mean(np.abs(shap_values), axis=0)
    }).sort_values('importance_score', ascending=False)

    importance_df.to_csv(os.path.join(PLOT_DIR, "feature_importance_ranking.csv"), index=False)

    df_res.to_csv(os.path.join(PLOT_DIR, "model_evaluation_results.csv"), index=False)

    print(f"\n✓ Saved detailed results to {PLOT_DIR}/")
    print(f"  - shap_detailed_results.csv")
    print(f"  - feature_importance_ranking.csv")
    print(f"  - model_evaluation_results.csv")

    print(f"\n🎉 SHAP Analysis Complete!")
    print(f" Generated {len(glob.glob(os.path.join(PLOT_DIR, '*.png')))} visualizations")
    print(f" All results saved in: {PLOT_DIR}/")

if __name__ == "__main__":
    try:
        run_comprehensive_shap_analysis()
    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        import traceback
        traceback.print_exc()
