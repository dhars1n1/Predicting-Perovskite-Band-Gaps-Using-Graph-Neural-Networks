# Data-Driven Material Modeling (DDMM)

This repository is a codebase for predicting perovskite material properties (band gap) using Graph Neural Networks (GNNs). This repository contains the full pipeline from raw CSV data to ranked candidate perovskites ready for downstream analysis.

Summary of concepts
- Perovskites: crystalline materials with general formula ABX3 (A/B = cations, X = anion). Properties depend on composition and site roles.
- Graph Neural Networks (GNNs): models that operate on graph-structured data. For perovskites we represent sites (A, B, X/additives) as nodes and the physical/chemical relationships as edges.
- Feature engineering: frequency encoding, one-hot site encodings, and scalar descriptors are used to construct node features that the GNN can consume.
- Objective: regress material properties (e.g., band gap) and rank candidates for applications (solar cells, LEDs).

Quickstart
1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# Windows: venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Preprocess data:
```bash
python data_preprocessing.py
```
4. Train the GNN:
```bash
python GNN.py
```
5. Predict & rank perovskites:
```bash
python pick_best_perovskite.py
```

Repository layout
- data/                      Raw and processed CSVs
- models/                    Saved model checkpoints (.pt/.pth)
- results/                   Logs and prediction outputs (.csv)
- data_preprocessing.py      Data cleaning & feature engineering
- GNN.py                     Dataset, model, and training loop
- pick_best_perovskite.py    Inference and candidate ranking
- requirements.txt           Python dependencies
- README.md                  This file

Detailed workflow and components

1) Data and preprocessing
- Expected raw input: data/perovskite_data.csv
  - Typical columns: composition (string), site columns (A_site, B_site, X_site or similar), numeric target (e.g., band_gap), optional metadata.
- Steps performed:
  - Basic cleaning: select relevant columns, drop duplicates, drop rows with missing critical fields.
  - Extract unique species: parse composition and site columns to enumerate distinct atomic species.
  - Categorization: label species as A-site, B-site, X-site, or additive/other using predefined sets and heuristics.
  - Frequency encoding:
    - Global: compute occurrence frequency for every species across dataset and replace species names with frequencies.
    - By-site: compute frequencies per site type (A, B, X) to capture site-specific prevalence.
  - Outputs (saved CSVs):
    - data/perovskite_filtered.csv
    - data/unique_species.csv
    - data/species_summary.csv
    - data/perovskite_frequency_encoded.csv
    - data/perovskite_frequency_encoded_by_site.csv

Notes: The preprocessing step is the place to adapt column names or add extra scalar descriptors (ionic radius, electronegativity) if available.

2) Graph construction
- Node definition:
  - Each sample becomes a small graph with nodes representing sites present in the sample (A, B, X sites; optionally additives).
  - Node features typically include:
    - Frequency-encoded value(s) for that species.
    - One-hot encoding of site type (A/B/X).
    - Optional scalar descriptors (computed or looked up).
- Edge definition:
  - For perovskites a fully-connected motif among sites is typical (i.e., edges between A-B, A-X, B-X).
  - Edge attributes are optional; if present they can encode bond-type proxies or site-distance features.
- Batching:
  - Graphs are collated using PyTorch Geometric Data and DataLoader for minibatch training.

3) Model architecture (GNN.py)
- Configurable components:
  - Convolution type: GCNConv or SAGEConv (selectable via conv_type).
  - Number of layers, hidden dimension, dropout rate.
- Typical structure:
  - Multiple graph convolution layers with nonlinearity and optional batchnorm.
  - Global pooling (global_mean_pool) to get graph-level embedding.
  - MLP head to regress the scalar target.
- Example signature:
```python
class GNN(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=3, dropout=0.2, conv_type="gcn")
```

4) Training loop and evaluation
- Loss: Mean Squared Error (MSE) for regression tasks.
- Optimizer: Adam (learning rate configurable).
- Validation: periodic evaluation on a held-out validation set; best checkpoint saved by validation metric (lowest val MSE).
- Early stopping: stop training when validation loss does not improve for N epochs.
- Metrics:
  - MSE, RMSE, MAE, and optionally R^2.
  - Log metrics per epoch to results/results_log.csv for experiment records.
- Data splits:
  - Train / Val / Test splits; ensure reproducible splits by setting seeds.
- Hyperparameter search:
  - Grid search over learning rate, batch size, number of layers, hidden dim, conv type. Record results and best config.

5) Inference and ranking (pick_best_perovskite.py)
- Load the saved best model and any saved scaler/normalizer.
- Convert rows from frequency-encoded CSV into graph Data objects using the same feature logic.
- Run batched inference, inverse-transform predictions (if scaling used).
- Save results to results/predicted_perovskites.csv and create ranked lists (e.g., top-5 for solar application).
- Output includes model prediction, sample identifier, and metadata required for downstream selection.

Engineering details and best practices
- Feature hygiene: ensure the same preprocessing step is used for training and inference. Save encoders (frequency maps, scalers).
- Reproducibility:
  - Set seeds (random, numpy, torch) and fix deterministic flags where possible.
  - Save training config (hyperparameters and git commit hash) with each run.
- Experiment tracking:
  - Log parameters and metrics in results/results_log.csv; consider adding MLflow or simple JSON logs per run.
- GPU usage:
  - Code checks for CUDA availability; set device explicitly if needed.
  - If GPU OOM occurs, reduce batch size or model size.
- Testing:
  - Add unit tests for parsing/composition splitting and graph creation to avoid regressions.

Extending the codebase
- Add CLI/config: move hyperparameters to a config file or CLI flags for reproducibility and easier experiments.
- Additional features:
  - Add lookup tables for physical descriptors (ionic radius, electronegativity) to enrich node features.
  - Incorporate edge attributes that reflect geometric/bonding approximations if structural data is available.
- Model improvements:
  - Try attention-based GNNs (GAT), message-passing networks, and ensembling to improve predictive performance.
- Deployment:
  - For production inference, create a minimal API wrapper that loads the saved model and runs inference on incoming compositions.

Files and outputs (concise)
- Inputs:
  - data/perovskite_data.csv (raw)
- Preprocessing outputs:
  - data/*.csv (filtered, encoded, summary)
- Models:
  - models/best_model.pt
- Results:
  - results/results_log.csv
  - results/predicted_perovskites.csv

Troubleshooting (common issues)
- Import errors: confirm virtualenv and requirements installed.
- Column mismatch: verify CSV header names; adapt data_preprocessing.py or rename CSV headers.
- Non-deterministic results: set seeds and log random states.
- Poor performance: check data leakage, insufficient features, class imbalance, or inadequate model capacity.

References and resources
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- Papers on GNNs in materials science: add project-specific citations here if needed.

Contributing and license
- Contributions: open issues or PRs; include tests when adding preprocessing or graph generation logic.
- License: MIT