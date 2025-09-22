# GNNExplainer for Perovskite Band Gap Prediction

This module implements a comprehensive Graph Neural Network (GNN) training pipeline with explainability analysis for predicting perovskite band gaps. The code performs hyperparameter optimization, trains multiple GNN models, and uses GNNExplainer to provide interpretable explanations of the model's predictions.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data Structure](#data-structure)
- [Code Architecture](#code-architecture)
- [Detailed Code Explanation](#detailed-code-explanation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Visualization](#visualization)

## Overview

The GNNExplainer module performs the following key tasks:
1. **Data Loading and Preprocessing**: Loads perovskite data and converts it into graph representations
2. **Graph Construction**: Creates molecular graphs where nodes represent atomic sites (A, B, C) in perovskite structures
3. **Hyperparameter Search**: Systematically tests different combinations of batch sizes, learning rates, epochs, and convolution types
4. **Model Training**: Trains GNN models using both GCN and GraphSAGE convolutions
5. **Model Evaluation**: Evaluates models using RMSE on validation and test sets
6. **Explainability Analysis**: Uses GNNExplainer to interpret model predictions and visualize important features

## Dependencies

```python
import sys, os
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
from torch_geometric.explain import Explainer, GNNExplainer
import itertools
```

## Data Structure

### Input Data Format
The code expects a CSV file with the following structure:
- **A_*** columns: Features for A-site atoms in perovskite (e.g., A_electronegativity, A_ionic_radius)
- **B_*** columns: Features for B-site atoms in perovskite
- **C_*** columns: Features for C-site atoms in perovskite (typically oxygen)
- **Perovskite_band_gap**: Target variable (band gap in eV)

### Graph Representation
Each perovskite compound is converted into a graph with:
- **3 nodes**: Representing A-site, B-site, and C-site atoms
- **Node features**: Concatenation of elemental features + one-hot encoding for site type
- **Edges**: Fully connected graph with 12 directed edges (bidirectional connections between all pairs)

## Code Architecture

### 1. Configuration and Setup
```python
CSV_PATH = "../data/perovskite_frequency_encoded_by_site.csv"
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience = 15
```

### 2. Data Loading and Cleaning
- Loads CSV data and validates required columns
- Cleans band gap values (handles ranges and units)
- Extracts A-site, B-site, and C-site features

### 3. Graph Construction Function
```python
def row_to_graph(row):
    # Extracts and processes features for each site
    # Creates node feature matrix with padding
    # Adds one-hot encoding for site types
    # Defines edge connectivity
    # Returns PyTorch Geometric Data object
```

### 4. GNN Model Architecture
```python
class GNN2(nn.Module):
    def __init__(self, in_dim, hidden=128, num_layers=3, dropout=0.2, conv_type="gcn"):
        # Configurable convolution layers (GCN or GraphSAGE)
        # Global mean pooling for graph-level representation
        # MLP head for regression
```

**Model Components:**
- **Convolution Layers**: Either GCN or GraphSAGE convolutions
- **Pooling**: Global mean pooling to aggregate node features
- **MLP Head**: Two-layer neural network for final prediction
- **Activation**: ReLU activations between layers
- **Regularization**: Dropout for preventing overfitting

### 5. Hyperparameter Search Grid
```python
hyperparams_grid = {
    "BATCH_SIZE": [16, 32],
    "LR": [1e-3, 5e-4],
    "EPOCHS": [50, 100],
    "CONV_TYPE": ["gcn", "sage"]
}
```
Total combinations: 2 × 2 × 2 × 2 = 16 different configurations

## Detailed Code Explanation

### Data Preprocessing

#### Band Gap Cleaning
```python
def clean_band_gap(val):
    val = str(val).replace("eV", "").strip()
    if "-" in val:
        # Handles range values by taking average
        parts = val.split("-")
        nums = [float(p) for p in parts if p.strip() != ""]
        return sum(nums) / len(nums) if nums else None
    return float(val)
```

#### Feature Processing
1. **Feature Extraction**: Separates columns by site type (A_, B_, C_ prefixes)
2. **Numeric Conversion**: Converts all features to float, filling NaN with 0
3. **Feature Padding**: Ensures all sites have the same feature dimensionality
4. **One-hot Encoding**: Adds site type information ([1,0,0], [0,1,0], [0,0,1])

### Graph Construction

#### Node Features
- **Raw Features**: Element-specific properties (electronegativity, radius, etc.)
- **Site Encoding**: One-hot vectors indicating A, B, or C site
- **Padding**: Ensures consistent dimensionality across all graphs

#### Edge Structure
```python
edge_index = torch.tensor([
    [0,1,1,0,0,2,2,0,1,2,2,1],  # Source nodes
    [1,0,0,1,2,0,0,2,2,0,1,1]   # Target nodes
], dtype=torch.long)
```
Creates bidirectional connections: A↔B, A↔C, B↔C

### Training Pipeline

#### Data Splitting
- **Training**: 70% of data
- **Validation**: 15% of data  
- **Test**: 15% of data
- **Reproducibility**: Fixed random seed, saved indices

#### Target Scaling
```python
y_train = np.array([d.y.item() for d in train_list]).reshape(-1,1)
scaler = StandardScaler().fit(y_train)
```
Standardizes band gap values using training set statistics

#### Training Loop
1. **Forward Pass**: Model prediction on batch
2. **Loss Calculation**: MSE between prediction and target
3. **Backpropagation**: Gradient computation and parameter update
4. **Validation**: Evaluate on validation set
5. **Early Stopping**: Stop if validation doesn't improve for 15 epochs
6. **Model Saving**: Save best model based on validation RMSE

### Model Evaluation
```python
def evaluate(loader, model, scaler):
    # Sets model to evaluation mode
    # Iterates through data loader
    # Computes predictions without gradients
    # Inverse transforms to original scale
    # Calculates RMSE in original units
```

### Explainability Analysis

#### Model Wrapper
```python
class ExplainerModelWrapper(nn.Module):
    def forward(self, x, edge_index, batch=None):
        # Adapts model interface for explainer compatibility
        # Handles batch dimension for single graphs
```

#### GNNExplainer Setup
```python
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=ModelConfig(mode='regression', task_level='graph', return_type='raw'),
)
```

**Explanation Components:**
- **Node Importance**: Which atomic sites are most important for prediction
- **Edge Importance**: Which connections between sites matter most
- **Feature Attribution**: How individual features contribute to predictions

#### Visualization Function
```python
def run_explainer_multi(model, test_list, scaler, conv_type, bs, lr, epochs,
                        explainer_epochs=200, seed=42, num_samples=5):
```

**Visualization Features:**
- **Multiple Samples**: Shows explanations for 5 random test cases
- **Triangular Layout**: Positions nodes in perovskite-like arrangement
- **Color Coding**: 
  - Node colors indicate importance (viridis colormap)
  - Edge colors show connection importance (magma colormap)
- **Directed Edges**: Shows directional information flow
- **Colorbars**: Quantitative importance scales
- **Plot Saving**: Saves high-resolution PNG with descriptive filename

### File Management

#### Directory Structure
```
GNNexplainer/
├── GNNexplainer.py          # Main script
├── README.md                # This file
└── gnn_explanation_*.png    # Generated plots

../data/
└── perovskite_frequency_encoded_by_site.csv

../models/
└── model_bs*_lr*_ep*_*.pth  # Trained models

../results/
├── results_log.csv          # Hyperparameter results
├── train_idx.npy           # Training indices
├── val_idx.npy             # Validation indices
└── test_idx.npy            # Test indices
```

#### Model Checkpoints
Each saved model contains:
- **model_state**: PyTorch state dictionary
- **scaler**: StandardScaler for target normalization
- **in_dim**: Input feature dimensionality
- **conv_type**: Convolution type used ("gcn" or "sage")

#### Results Logging
CSV format: `batch_size,learning_rate,epochs,conv_type,best_val_rmse,test_rmse`

## Usage

### Basic Execution
```bash
cd GNNexplainer
python GNNexplainer.py
```

### Custom Data Path
```bash
python GNNexplainer.py path/to/your/data.csv
```

### Expected Runtime
- **Total Training**: ~30-60 minutes (16 hyperparameter combinations)
- **Per Configuration**: 2-5 minutes depending on early stopping
- **Explanation Generation**: 2-3 minutes for 5 samples

## Output Files

### 1. Model Files
- **Location**: `../models/`
- **Naming**: `model_bs{batch_size}_lr{lr}_ep{epochs}_{conv_type}.pth`
- **Content**: Complete model checkpoint for reproducibility

### 2. Results Log
- **Location**: `../results/results_log.csv`
- **Content**: Performance metrics for all hyperparameter combinations
- **Format**: CSV with columns for parameters and RMSE scores

### 3. Data Splits
- **Location**: `../results/`
- **Files**: `train_idx.npy`, `val_idx.npy`, `test_idx.npy`
- **Purpose**: Ensure reproducible train/validation/test splits

### 4. Explanation Plots
- **Location**: `GNNexplainer/`
- **Naming**: `gnn_explanation_{conv_type}_bs{bs}_lr{lr}_ep{epochs}.png`
- **Resolution**: 300 DPI for publication quality
- **Content**: Node and edge importance visualizations

## Visualization

### Plot Interpretation

#### Node Colors (Viridis Scale)
- **Dark Purple**: Low importance sites
- **Blue/Green**: Medium importance sites  
- **Yellow**: High importance sites

#### Edge Colors (Magma Scale)
- **Dark Purple**: Weak connections
- **Red/Orange**: Medium strength connections
- **Yellow/White**: Strong, important connections

#### Layout
- **Triangular Arrangement**: Mimics perovskite ABO₃ structure
- **A-site**: Top vertex
- **B-site**: Bottom left vertex
- **C-site**: Bottom right vertex

### Scientific Insights
The explanations can reveal:
1. **Dominant Sites**: Which atomic positions most influence band gap
2. **Critical Interactions**: Important chemical bonding patterns
3. **Feature Importance**: Which elemental properties matter most
4. **Structure-Property Relations**: How atomic arrangement affects electronic properties

## Technical Notes

### Memory Considerations
- **GPU Usage**: Automatically detects and uses CUDA if available
- **Batch Processing**: Configurable batch sizes to manage memory
- **Graph Size**: Small graphs (3 nodes) minimize memory requirements

### Reproducibility
- **Fixed Seeds**: NumPy and PyTorch seeds set for reproducible results
- **Saved Indices**: Train/test splits preserved across runs
- **Deterministic Operations**: Consistent results across executions

### Error Handling
- **Path Validation**: Automatic directory creation
- **Data Validation**: Checks for required columns and valid values
- **Gradient Issues**: Robust color mapping prevents plotting errors

### Performance Optimization
- **Early Stopping**: Prevents overfitting and reduces training time
- **Efficient Evaluation**: Batch processing for faster validation
- **Memory Management**: Proper cleanup and device management

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure data file exists at specified path
   - Check directory permissions for output folders

2. **CUDA Errors**
   - Reduce batch size if out of memory
   - Check GPU availability and PyTorch installation

3. **Plot Generation Issues**
   - Verify matplotlib backend compatibility
   - Check write permissions in GNNexplainer directory

4. **Import Errors**
   - Install required packages: `pip install torch torch-geometric matplotlib networkx scikit-learn pandas numpy`

### Performance Tips

1. **Faster Training**
   - Reduce hyperparameter grid size
   - Use smaller epoch counts for initial testing
   - Increase patience for early stopping

2. **Better Explanations**
   - Increase `explainer_epochs` for more stable explanations
   - Test with more samples using `num_samples` parameter
   - Experiment with different explanation algorithms

This comprehensive implementation provides both predictive modeling and interpretability analysis for perovskite band gap prediction, making it valuable for both machine learning research and materials science applications.