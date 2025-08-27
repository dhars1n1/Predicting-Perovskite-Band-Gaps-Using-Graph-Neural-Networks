# Robust chemistry-aware GNN with EDGE FEATURES (NaN-safe, flexible parsing)

import ast, math, random, re
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from pymatgen.core.periodic_table import Element
import os
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
INPUT_CSV = "../data/perovskite_numeric_encoded.csv"   # set your input file

# column names 
COL_REF_ID = "Ref_ID"
COL_DOI = "Ref_DOI_number"
COL_SINGLE_CRYSTAL = "Perovskite_single_crystal"
COL_DIM_3D_2D = "Perovskite_dimension_3D_with_2D_capping_layer"
COL_DIM_LIST = "Perovskite_dimension_list_of_layers"
COL_ABC3 = "Perovskite_composition_perovskite_ABC3_structure"
COL_INSPIRED = "Perovskite_composition_perovskite_inspired_structure"
COL_A_IONS = "Perovskite_composition_a_ions"
COL_A_COEFF = "Perovskite_composition_a_ions_coefficients"
COL_B_IONS = "Perovskite_composition_b_ions"
COL_B_COEFF = "Perovskite_composition_b_ions_coefficients"
COL_C_IONS = "Perovskite_composition_c_ions"
COL_C_COEFF = "Perovskite_composition_c_ions_coefficients"
COL_SHORT = "Perovskite_composition_short_form"
COL_LONG = "Perovskite_composition_long_form"
COL_INORGANIC = "Perovskite_composition_inorganic"
COL_LEADFREE = "Perovskite_composition_leadfree"
COL_ADD_COMPS = "Perovskite_additives_compounds"
COL_ADD_CONC = "Perovskite_additives_concentrations"
BANDGAP_COL = "Perovskite_band_gap"
BANDGAP_GRADED = "Perovskite_band_gap_graded"
PL_MAX = "Perovskite_pl_max"
COL_DIM_COMBINED = "Perovskite_dimension_combined"
TMIN_COL = "Stability_temperature_min"
TMAX_COL = "Stability_temperature_max"

# dict-like columns if available
PARSED_COL = "parsed_composition"
FRACTION_COL = "composition_fraction"

# training
RANDOM_SEED = 42
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 80
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------- organic and fallback data ----------
ORGANIC = {
    "MA": {"radius": 2.70, "mass": 32.0},
    "FA": {"radius": 2.79, "mass": 45.0},
    "GA": {"radius": 2.78, "mass": 59.0},
    "PEA": {"radius": 3.10, "mass": 122.0},
}

# ---------- parsing helpers ----------
SEP_RE = re.compile(r"[;,\|\s/]+")

def split_flexible(s: str) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none", "unknown"}:
        return []
    toks = [t.strip() for t in SEP_RE.split(s) if t.strip() != ""]
    # remove numeric-only tokens
    toks = [t for t in toks if not re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)]
    return toks

def parse_numeric_list(s: str) -> List[float]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    toks = split_flexible(s)
    out = []
    for t in toks:
        try:
            out.append(float(t))
        except:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
            if m:
                try:
                    out.append(float(m.group(0)))
                except:
                    pass
    return out

def parse_range_avg(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if s.lower() in {"nan", "none", "unknown", ""}:
        return np.nan
    try:
        return float(s)
    except:
        pass
    toks = re.split(r"[-–—]|;|,|\s+to\s+|\s+", s)
    nums = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        try:
            nums.append(float(t))
        except:
            m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
            for mm in m:
                try:
                    nums.append(float(mm))
                except:
                    pass
    if len(nums) == 0:
        return np.nan
    return float(np.mean(nums))

def safe_literal_eval(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        return ast.literal_eval(str(s))
    except:
        t = str(s).replace(";", ",")
        if ":" in t and "{" not in t:
            t2 = "{" + t + "}"
            try:
                return ast.literal_eval(t2)
            except:
                pass
        try:
            return ast.literal_eval(t)
        except:
            return None

# ---------- element property helpers ----------
def get_site_radius(el: str, site_hint: str = "A") -> float:
    if el in ORGANIC:
        return float(ORGANIC[el]["radius"])
    try:
        e = Element(el)
        ox = {"A": +1, "B": +2, "X": -1}.get(site_hint)
        if ox is not None:
            r = e.ionic_radii.get(ox, None)
            if r is not None:
                return float(r)
        if getattr(e, "average_ionic_radius", None) is not None:
            return float(e.average_ionic_radius)
    except:
        pass
    return np.nan

def get_element_numeric(el: str) -> Tuple[float,float,float]:
    if el in ORGANIC:
        return np.nan, float(ORGANIC[el]["mass"]), np.nan
    try:
        e = Element(el)
        Z = float(e.Z) if e.Z is not None else np.nan
        mass = float(e.atomic_mass) if e.atomic_mass is not None else np.nan
        eneg = float(e.X) if e.X is not None else np.nan
        return Z, mass, eneg
    except:
        return np.nan, np.nan, np.nan

def weighted_site_radius(elements: List[str], coeffs: List[float], site: str, fallback_fracs: Dict[str,float]):
    if not elements:
        return np.nan
    if not coeffs or len(coeffs) != len(elements):
        coeffs = [fallback_fracs.get(el, 1.0) for el in elements]
    vals, ws = [], []
    for el, w in zip(elements, coeffs):
        r = get_site_radius(el, site)
        if not np.isnan(r):
            vals.append(r); ws.append(float(w))
    if not vals:
        return np.nan
    total = sum(ws) if sum(ws) != 0 else 1.0
    return float(sum(v * (w/total) for v,w in zip(vals, ws)))

# ---------- EDGE FEATURE BUILDER ----------
# Directed edge type one-hots for (srcSite -> dstSite): A,B,X × A,B,X = 9
EDGE_SITE_TYPES = {("A","A"):0, ("A","B"):1, ("A","X"):2,
                   ("B","A"):3, ("B","B"):4, ("B","X"):5,
                   ("X","A"):6, ("X","B"):7, ("X","X"):8}
N_EDGE_SITE_TYPES = 9

def site_of(el: str, A_list, B_list, X_list) -> str:
    if el in A_list: return "A"
    if el in B_list: return "B"
    if el in X_list: return "X"
    return "A"  # default fallback

def make_edge_features(node_feats: np.ndarray,
                       node_syms: List[str],
                       A_list: List[str], B_list: List[str], X_list: List[str]) -> np.ndarray:
    """
    node_feats columns (9): [Z, mass, eneg, radius, frac, is_org, is_A, is_B, is_X]
    Build directed edges i->j (i!=j) with features:
      - one-hot(site_i->site_j) [9]
      - |Z_i - Z_j|
      - |eneg_i - eneg_j|
      - |radius_i - radius_j|
      - (radius_i + radius_j)
      - (frac_i * frac_j)
      - src_is_org, dst_is_org
    Total edge features: 9 + 6 + 2 = 17
    """
    n = node_feats.shape[0]
    if n < 2:
        return np.zeros((0, 17), dtype=np.float32)
    edges = []
    Z, mass, eneg, radius, frac, is_org = (node_feats[:,0], node_feats[:,1], node_feats[:,2],
                                           node_feats[:,3], node_feats[:,4], node_feats[:,5])
    for i in range(n):
        si = site_of(node_syms[i], A_list, B_list, X_list)
        for j in range(n):
            if i == j: continue
            sj = site_of(node_syms[j], A_list, B_list, X_list)
            onehot = np.zeros(N_EDGE_SITE_TYPES, dtype=np.float32)
            onehot[EDGE_SITE_TYPES[(si, sj)]] = 1.0
            feats = [
                abs(Z[i] - Z[j]) if not (np.isnan(Z[i]) or np.isnan(Z[j])) else 0.0,
                abs(eneg[i] - eneg[j]) if not (np.isnan(eneg[i]) or np.isnan(eneg[j])) else 0.0,
                abs(radius[i] - radius[j]) if not (np.isnan(radius[i]) or np.isnan(radius[j])) else 0.0,
                (0.0 if np.isnan(radius[i]) else radius[i]) + (0.0 if np.isnan(radius[j]) else radius[j]),
                (0.0 if np.isnan(frac[i]) else frac[i]) * (0.0 if np.isnan(frac[j]) else frac[j]),
                float(is_org[i]), float(is_org[j]),
            ]
            edges.append(np.concatenate([onehot, np.array(feats, dtype=np.float32)], axis=0))
    return np.vstack(edges).astype(np.float32)

# ---------- graph builder ----------
def build_graph(row: pd.Series):
    parsed = safe_literal_eval(row.get(PARSED_COL)) if PARSED_COL in row.index else None
    fracdict = safe_literal_eval(row.get(FRACTION_COL)) if FRACTION_COL in row.index else None

    if isinstance(fracdict, dict) and len(fracdict) > 0:
        fracs = {str(k): float(v) for k,v in fracdict.items()}
    elif isinstance(parsed, dict) and len(parsed) > 0:
        tmp = {str(k): float(v) for k,v in parsed.items()}
        s = sum(tmp.values()) if sum(tmp.values()) != 0 else 1.0
        fracs = {k: v/s for k,v in tmp.items()}
    else:
        longform = row.get(COL_LONG)
        lf = safe_literal_eval(longform)
        if isinstance(lf, dict) and len(lf) > 0:
            tmp = {str(k): float(v) for k,v in lf.items()}
            s = sum(tmp.values()) if sum(tmp.values()) != 0 else 1.0
            fracs = {k: v/s for k,v in tmp.items()}
        else:
            A_list = split_flexible(row.get(COL_A_IONS))
            B_list = split_flexible(row.get(COL_B_IONS))
            X_list = split_flexible(row.get(COL_C_IONS))
            elems = list(dict.fromkeys(A_list + B_list + X_list))
            if len(elems) == 0:
                return None
            fracs = {el: 1.0/len(elems) for el in elems}

    elements = list(fracs.keys())
    if len(elements) == 0:
        return None

    A_list = split_flexible(row.get(COL_A_IONS))
    B_list = split_flexible(row.get(COL_B_IONS))
    X_list = split_flexible(row.get(COL_C_IONS))

    a_coeffs = parse_numeric_list(row.get(COL_A_COEFF))
    b_coeffs = parse_numeric_list(row.get(COL_B_COEFF))
    x_coeffs = parse_numeric_list(row.get(COL_C_COEFF))

    node_feats = []
    node_syms = []
    for el in elements:
        Z, mass, eneg = get_element_numeric(el)
        site = "A" if el in A_list else ("B" if el in B_list else ("X" if el in X_list else "A"))
        radius = get_site_radius(el, site)
        frac = float(fracs.get(el, 0.0))
        is_org = 1.0 if el in ORGANIC else 0.0
        is_A = 1.0 if el in A_list else 0.0
        is_B = 1.0 if el in B_list else 0.0
        is_X = 1.0 if el in X_list else 0.0
        zf = 0.0 if pd.isna(Z) else float(Z)
        massf = 0.0 if pd.isna(mass) else float(mass)
        enegf = 0.0 if pd.isna(eneg) else float(eneg)
        radf = 0.0 if pd.isna(radius) else float(radius)
        node_feats.append([zf, massf, enegf, radf, frac, is_org, is_A, is_B, is_X])
        node_syms.append(el)

    x = torch.tensor(node_feats, dtype=torch.float)
    n = x.size(0)

    # Build edges: fully connected directed (no self-loops)
    if n >= 2:
        src, dst = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    src.append(i); dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # Edge features
        edge_attr_np = make_edge_features(x.numpy(), node_syms, A_list, B_list, X_list)
        edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0,17), dtype=torch.float)

    # Global features
    bandgap = parse_range_avg(row.get(BANDGAP_COL))
    tmin = parse_range_avg(row.get(TMIN_COL))
    tmax = parse_range_avg(row.get(TMAX_COL))
    if not np.isnan(tmin) and not np.isnan(tmax):
        tmean = 0.5*(tmin + tmax)
    elif not np.isnan(tmin):
        tmean = float(tmin)
    elif not np.isnan(tmax):
        tmean = float(tmax)
    else:
        tmean = np.nan

    def pick_coeffs(species, coeffs, fracs):
        if species and coeffs and len(coeffs) == len(species):
            return coeffs
        return [fracs.get(sp, 0.0) for sp in species] if species else []

    A_w = pick_coeffs(A_list, a_coeffs, fracs)
    B_w = pick_coeffs(B_list, b_coeffs, fracs)
    X_w = pick_coeffs(X_list, x_coeffs, fracs)

    rA = weighted_site_radius(A_list, A_w, "A", fracs) if A_list else np.nan
    rB = weighted_site_radius(B_list, B_w, "B", fracs) if B_list else np.nan
    rX = weighted_site_radius(X_list, X_w, "X", fracs) if X_list else np.nan
    if not (np.isnan(rA) or np.isnan(rB) or np.isnan(rX)):
        t_factor = (rA + rX) / (math.sqrt(2) * (rB + rX)) if (rB + rX) != 0 else np.nan
        mu_factor = (rB / rX) if rX != 0 else np.nan
    else:
        t_factor = np.nan; mu_factor = np.nan

    # Target: proxy energy = bandgap * tmean
    if np.isnan(bandgap) or np.isnan(tmean):
        return None
    E_proxy = float(bandgap) * float(tmean)

    gf = torch.tensor([[0.0 if np.isnan(bandgap) else float(bandgap),
                        0.0 if np.isnan(tmean) else float(tmean),
                        0.0 if np.isnan(t_factor) else float(t_factor),
                        0.0 if np.isnan(mu_factor) else float(mu_factor)]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([E_proxy], dtype=torch.float))
    data.global_feats = gf
    data.node_symbols = node_syms
    data.ref_id = row.get(COL_REF_ID)
    data.ref_doi = row.get(COL_DOI)
    return data

# ---------- load CSV and build graphs ----------
print("Loading CSV:", INPUT_CSV)
raw = pd.read_csv(INPUT_CSV, low_memory=False)
graphs = []
skipped = 0
for _, row in raw.iterrows():
    try:
        g = build_graph(row)
    except Exception:
        g = None
    if g is None:
        skipped += 1
        continue
    graphs.append(g)
print(f"Built {len(graphs)} graphs; skipped {skipped} rows due to missing/invalid data.")

if len(graphs) == 0:
    raise RuntimeError("No usable graphs — check input for bandgap and temperature availability.")

# ---------- SCALE FEATURES (node, edge, global) ----------
# Node feature scaling
all_nodes = np.vstack([g.x.numpy() for g in graphs])
node_mean = all_nodes.mean(axis=0, keepdims=True)
node_std = all_nodes.std(axis=0, keepdims=True) + 1e-8
for g in graphs:
    arr = g.x.numpy()
    arr = (arr - node_mean) / node_std
    g.x = torch.tensor(arr, dtype=torch.float)

# Edge feature scaling
# Concatenate all edge_attr rows (may be empty for some graphs)
edge_rows = [g.edge_attr.numpy() for g in graphs if g.edge_attr.numel() > 0]
if len(edge_rows) > 0:
    all_edges = np.vstack(edge_rows)
    edge_mean = all_edges.mean(axis=0, keepdims=True)
    edge_std = all_edges.std(axis=0, keepdims=True) + 1e-8
else:
    # fallback for rare degenerate case
    edge_mean = np.zeros((1,17), dtype=np.float32)
    edge_std = np.ones((1,17), dtype=np.float32)
for g in graphs:
    if g.edge_attr.numel() > 0:
        ea = g.edge_attr.numpy()
        ea = (ea - edge_mean) / edge_std
        g.edge_attr = torch.tensor(ea, dtype=torch.float)

# Global features scaling
globals_mat = torch.vstack([g.global_feats for g in graphs])  # [N,1,4] if kept as (1,F)
globals_mat = globals_mat.squeeze(1)  # [N,4]
glob_mean = globals_mat.mean(dim=0, keepdim=True)
glob_std = globals_mat.std(dim=0, keepdim=True) + 1e-8
globals_scaled = (globals_mat - glob_mean) / glob_std
for i,g in enumerate(graphs):
    g.global_feats = globals_scaled[i].unsqueeze(0)

# ---------- split dataset ----------
N = len(graphs)
idxs = list(range(N))
random.shuffle(idxs)
ntrain = max(1, int(0.8 * N))
nval = max(1, int(0.1 * N))
train_idx = idxs[:ntrain]
val_idx = idxs[ntrain:ntrain+nval]
test_idx = idxs[ntrain+nval:]

train_graphs = [graphs[i] for i in train_idx]
val_graphs = [graphs[i] for i in val_idx]
test_graphs = [graphs[i] for i in test_idx]

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train {len(train_graphs)} | Val {len(val_graphs)} | Test {len(test_graphs)}")

# ---------- compute target normalization from TRAIN only ----------
y_train = np.concatenate([g.y.numpy() for g in train_graphs]).astype(np.float32)
y_mean = float(y_train.mean())
y_std = float(y_train.std() + 1e-8)  # avoid divide-by-zero

print(f"Target normalization: mean={y_mean:.4f}, std={y_std:.4f}")

# ---------- MODEL ----------
class ChemGNN_Edge(nn.Module):
    def __init__(self, node_in, edge_in, global_in, hidden=64, layers=3, dropout=0.1):
        super().__init__()
        # Edge-aware GINEConv layers
        # Message MLP for node features
        self.msg_nn = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # First layer with edge_dim
        self.convs.append(GINEConv(self.msg_nn, edge_dim=edge_in))
        self.bns.append(nn.BatchNorm1d(hidden))
        # Additional layers
        for _ in range(layers-1):
            nn_layer = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(nn_layer, edge_dim=edge_in))
            self.bns.append(nn.BatchNorm1d(hidden))

        # Readout head
        self.mlp = nn.Sequential(
            nn.Linear(hidden + global_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
        hg = global_mean_pool(x, batch)  # [B, hidden]
        gf = data.global_feats.to(hg.device)  # [B, global_in]
        if gf.dim() == 1:
            gf = gf.unsqueeze(1)
        out = self.mlp(torch.cat([hg, gf], dim=1)).squeeze(1)
        return out

node_in = graphs[0].x.shape[1]
edge_in = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr.numel() > 0 else 17
global_in = graphs[0].global_feats.shape[1]
model = ChemGNN_Edge(
    node_in=node_in, edge_in=edge_in, global_in=global_in,
    hidden=128, layers=4, dropout=0.2
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)  # a bit more WD
loss_fn = nn.SmoothL1Loss()  # Huber-like, more robust than MSE
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.7, patience=8, verbose=True
)

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# ---------- TRAINING ----------
os.makedirs("plots", exist_ok=True)

best_val = float("inf")
best_state = None
patience = 0

train_loss_hist, val_mae_hist, val_rmse_hist = [], [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for batch in train_loader:
        batch = batch.to(DEVICE)
        # skip bad labels
        if torch.any(torch.isnan(batch.y)) or torch.any(torch.isinf(batch.y)):
            continue

        opt.zero_grad()
        pred = model(batch)

        # normalize prediction and target with TRAIN stats
        pred_n = (pred - y_mean) / y_std
        y_n = (batch.y - y_mean) / y_std

        loss = loss_fn(pred_n, y_n)
        if torch.isnan(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # gradient clipping
        opt.step()
        train_losses.append(loss.item())

    train_loss = np.nan if len(train_losses) == 0 else float(np.mean(train_losses))

    # ---------- validation ----------
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            pred = model(batch)
            # compute metrics in ORIGINAL scale
            val_preds.append(pred.cpu().numpy())
            val_targets.append(batch.y.cpu().numpy())

    if val_preds:
        y = np.concatenate(val_targets)
        p = np.concatenate(val_preds)
        val_mae = float(np.mean(np.abs(y - p)))
        val_rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    else:
        val_mae = val_rmse = float("nan")

    # history + logs
    train_loss_hist.append(train_loss)
    val_mae_hist.append(val_mae)
    val_rmse_hist.append(val_rmse)
    print(f"Epoch {epoch:03d} | TrainLoss(n) {train_loss:.6f} | Val MAE {val_mae:.6f} | Val RMSE {val_rmse:.6f}")

    # LR scheduler on validation RMSE
    if not math.isnan(val_rmse):
        scheduler.step(val_rmse)

    # early stopping on RMSE
    if not math.isnan(val_rmse) and val_rmse < best_val - 1e-8:
        best_val = val_rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
        # optional: save checkpoint
        torch.save({"model": best_state, "y_mean": y_mean, "y_std": y_std}, "plots/best_checkpoint.pt")
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print("Early stopping")
            break

# ---------- save plots ----------
plt.figure(figsize=(8, 6))
plt.plot(train_loss_hist, label="Train Loss (normalized)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss (Normalized y)")
plt.legend(); plt.savefig("plots/train_loss_final.png"); plt.close()

plt.figure(figsize=(8, 6))
plt.plot(val_mae_hist, label="Val MAE")
plt.plot(val_rmse_hist, label="Val RMSE")
plt.xlabel("Epoch"); plt.ylabel("Error"); plt.title("Validation Metrics (Original Scale)")
plt.legend(); plt.savefig("plots/val_metrics_final.png"); plt.close()

# load best
if best_state:
    model.load_state_dict(best_state)


# ---------- TEST ----------
model.eval()
test_preds = []; test_targets = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        p = model(batch)
        test_preds.append(p.cpu().numpy()); test_targets.append(batch.y.cpu().numpy())
if test_preds:
    y = np.concatenate(test_targets); p = np.concatenate(test_preds)
    test_mae = float(np.mean(np.abs(y-p))); test_rmse = float(np.sqrt(np.mean((y-p)**2)))
    print(f"Test MAE {test_mae:.6f} | Test RMSE {test_rmse:.6f}")
else:
    print("No test data")
