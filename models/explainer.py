import os
import torch
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain import ModelConfig

from .gnn import ChemGNN_Edge 


def _safe_resize(arr, length):
    arr = np.array(arr).ravel()
    if len(arr) >= length:
        return arr[:length]
    else:
        return np.pad(arr, (0, length - len(arr)), constant_values=0)


def flatten_features(graphs, expected_dim=None):
    flattened = []

    # First pass: compute maximum feature length if expected_dim not provided
    if expected_dim is None:
        max_len = 0
        for g in graphs:
            feats = []
            if hasattr(g, "x") and g.x is not None:
                feats.extend(g.x.mean(dim=0).cpu().numpy().ravel())
            if hasattr(g, "edge_attr") and g.edge_attr is not None:
                feats.extend(g.edge_attr.mean(dim=0).cpu().numpy().ravel())
            if hasattr(g, "global_feats") and g.global_feats is not None:
                # flatten even nested lists
                gf = np.array(g.global_feats)
                feats.extend(gf.ravel() if gf.ndim > 0 else [gf])
            max_len = max(max_len, len(feats))
        expected_dim = max_len

    # Second pass: flatten, pad/truncate
    for g in graphs:
        feats = []
        if hasattr(g, "x") and g.x is not None:
            feats.extend(g.x.mean(dim=0).cpu().numpy().ravel())
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            feats.extend(g.edge_attr.mean(dim=0).cpu().numpy().ravel())
        if hasattr(g, "global_feats") and g.global_feats is not None:
            gf = np.array(g.global_feats)
            feats.extend(gf.ravel() if gf.ndim > 0 else [gf])

        # Safe resize
        feats = np.array(feats, dtype=np.float32)
        if len(feats) < expected_dim:
            feats = np.pad(feats, (0, expected_dim - len(feats)), constant_values=0)
        else:
            feats = feats[:expected_dim]

        flattened.append(feats)

    return np.vstack(flattened)




def load_model_and_data(model_path, train_path, test_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Load graphs

    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    train_graphs = torch.load(os.path.join(plots_dir, "train_graphs.pt"), map_location=device)
    test_graphs = torch.load(os.path.join(plots_dir, "test_graphs.pt"), map_location=device)


    # Infer input dimensions from a sample graph
    g0 = train_graphs[0]
    node_in = g0.x.shape[1]
    edge_in = g0.edge_attr.shape[1] if (hasattr(g0, "edge_attr") and g0.edge_attr is not None) else 17
    global_in = g0.global_feats.shape[1] if hasattr(g0, "global_feats") else 0

    # Instantiate model
    model = ChemGNN_Edge(
        node_in=node_in,
        edge_in=edge_in,
        global_in=global_in,
        hidden=128,
        layers=4,
        dropout=0.2
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, train_graphs, test_graphs, checkpoint


def shap_analysis(model, train_graphs, test_graphs, checkpoint, out_dir, device):
    """
    Run SHAP analysis on ChemGNN_Edge model using global_feats only.
    Keeps node and edge features fixed, perturbs global features.
    """
    print("Running SHAP analysis...")

    os.makedirs(out_dir, exist_ok=True)

    # Use the global features from the training graphs as background
    background = []
    for g in train_graphs[:100]:  # use subset for efficiency
        if hasattr(g, "global_feats") and g.global_feats is not None:
            background.append(g.global_feats.cpu().numpy().ravel())
    background = np.array(background)
    gf_len = background.shape[1]

    # Prepare test features
    test_feats = []
    for g in test_graphs[:50]:
        if hasattr(g, "global_feats") and g.global_feats is not None:
            test_feats.append(g.global_feats.cpu().numpy().ravel())
    test_feats = np.array(test_feats)

    y_mean = checkpoint.get("y_mean", 0.0)
    y_std = checkpoint.get("y_std", 1.0)

    # Define SHAP model wrapper
    def model_predict(feats_array):
        preds = []
        for i, f in enumerate(feats_array):
            # Ensure correct length
            f = f[:gf_len]
            # Use corresponding graph or first graph if index out of range
            g = test_graphs[i] if i < len(test_graphs) else test_graphs[0]
            new_g = Data(
                x=g.x.to(device),
                edge_index=g.edge_index.to(device),
                edge_attr=g.edge_attr.to(device) if g.edge_attr is not None else None,
                global_feats=torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                p = model(new_g)
                if isinstance(p, torch.Tensor):
                    p = p.item()
                # Denormalize prediction
                p = p * y_std + y_mean
            preds.append(p)
        return np.array(preds)

    # Create SHAP explainer
    masker = shap.maskers.Independent(background, max_samples=100)
    explainer = shap.Explainer(model_predict, masker)
    shap_values = explainer(test_feats)

    # Convert to NumPy array
    shap_values_np = shap_values.values if hasattr(shap_values, "values") else shap_values

    # Plot summary
    shap.summary_plot(shap_values_np, test_feats, show=False)
    plt.savefig(os.path.join(out_dir, "shap_summary.png"))
    plt.close()

    return shap_values_np, test_feats



def gnn_explainer_analysis(model, graph, device, out_dir):
    print("Running GNNExplainer...")
    os.makedirs(out_dir, exist_ok=True)

    try:
        model_config = ModelConfig(
            mode="regression",
            task_level="graph",
            return_type="raw"
        )

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=50, return_type='regression'),
            model_config=model_config,
            explanation_type="model",
            edge_mask_type="object",
            node_mask_type="object",
        )
    except Exception as e:
        print(f"[WARN] Explainer instantiation failed: {e}")
        return

    try:
        # Pass node features and edge_index explicitly
        explanation = explainer(graph.to(device))


        # Visualize node feature importance
        explanation.visualize_feature_importance(
            path=os.path.join(out_dir, "node_feat_importance.png")
        )

        # Visualize subgraph for node 0
        explanation.visualize_subgraph(
            node_idx=0,
            path=os.path.join(out_dir, "explained_subgraph.png")
        )

    except Exception as e:
        print(f"[WARN] Explanation run failed: {e}")





def save_heatmap(shap_values, features, out_path):
    df = pd.DataFrame(shap_values, columns=[f"f{i}" for i in range(features.shape[1])])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0)
    plt.title("SHAP Value Correlations")
    plt.savefig(out_path)
    plt.close()


def dataset_summary(train_graphs, out_path):
    lengths = [g.num_nodes for g in train_graphs]
    df = pd.DataFrame({"num_nodes": lengths})
    plt.figure()
    sns.histplot(df["num_nodes"], bins=30)
    plt.title("Graph Size Distribution")
    plt.savefig(out_path)
    plt.close()


def run_hybrid_explanation_example(model_path, train_path, test_path, out_dir="ddmm/models/plots"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_graphs, test_graphs, checkpoint = load_model_and_data(model_path, train_path, test_path, device)

    shap_values, test_feats = shap_analysis(model, train_graphs, test_graphs, checkpoint, out_dir, device)
    save_heatmap(shap_values, test_feats, os.path.join(out_dir, "shap_heatmap.png"))
    dataset_summary(train_graphs, os.path.join(out_dir, "dataset_summary.png"))

    if test_graphs:
        gnn_explainer_analysis(model, test_graphs[0], device, out_dir)


if __name__ == "__main__":
    run_hybrid_explanation_example(
        "ddmm/models/plots/best_checkpoint.pt",
        "ddmm/models/plots/train_graphs.pt",
        "ddmm/models/plots/test_graphs.pt",
    )
