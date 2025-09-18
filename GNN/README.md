# Graph Neural Network (GNN) for Perovskite Band Gap Prediction

This document provides a detailed explanation of the `GNN.py` script, which implements a Graph Neural Network (GNN) to predict the band gap of perovskite materials. The script includes data preprocessing, graph construction, model architecture, training, and evaluation.

---

## Overview

The `GNN.py` script is designed to:
1. Load and preprocess perovskite data.
2. Represent each perovskite as a graph with nodes and edges.
3. Train a GNN model to predict the band gap of perovskites.
4. Evaluate the model using validation and test datasets.

---

## Input Data

### Input
The input data is a CSV file containing frequency-encoded features for A-site, B-site, and C-site atoms in perovskites. The file must include:
- **A-site features**: Columns prefixed with `A_`.
- **B-site features**: Columns prefixed with `B_`.
- **C-site features**: Columns prefixed with `C_`.
- **Target property**: `Perovskite_band_gap` (in eV).

### Input Dimensions
- **Node Feature Dimension**: Each node (A-site, B-site, C-site) has a feature vector consisting of:
  - Frequency-encoded values for the site.
  - One-hot encoding for the site type (A, B, or C).
- **Graph Input Dimension**: The concatenated feature vector for each node, including frequency-encoded values and one-hot encoding.

---

## GNN Architecture

The GNN model is implemented using PyTorch Geometric and consists of the following components:

### 1. **Message Passing (Graph Convolution Layers)**
- **Purpose**: Aggregates information from neighboring nodes to update node features.
- **Layers**: 
  - `GCNConv` (Graph Convolutional Network) or `SAGEConv` (GraphSAGE) layers.
  - The first layer maps the input node features to a hidden dimension.
  - Subsequent layers refine the hidden representation.
- **Output**: Updated node features after each layer.

### 2. **Global Pooling Layer**
- **Purpose**: Aggregates node-level features into a graph-level representation.
- **Method**: Global mean pooling.
- **Output**: A single vector representing the entire graph.

### 3. **MLP Head**
- **Purpose**: Maps the graph-level representation to the target property (band gap).
- **Structure**:
  - Fully connected layers.
  - ReLU activation.
  - Dropout for regularization.
- **Output**: Predicted band gap (scalar).

---

## Pipeline Inside GNN

### Step-by-Step Process
1. **Input Node Features**:
   - Each node is initialized with its feature vector.
2. **Message Passing**:
   - Node features are updated by aggregating information from neighboring nodes.
   - This is performed by the graph convolution layers.
3. **Global Pooling**:
   - Node features are pooled into a graph-level representation.
4. **MLP Head**:
   - The graph-level representation is passed through the MLP to predict the band gap.

### Layer Outputs
| Layer                  | Output Shape                     | Description                                  |
|------------------------|-----------------------------------|----------------------------------------------|
| Input Node Features    | `(num_nodes, node_feature_dim)`  | Initial node features.                      |
| Graph Convolution 1    | `(num_nodes, hidden_dim)`        | Updated node features after first layer.    |
| Graph Convolution N    | `(num_nodes, hidden_dim)`        | Updated node features after N layers.       |
| Global Pooling         | `(1, hidden_dim)`                | Graph-level representation.                 |
| MLP Head               | `(1, 1)`                         | Predicted band gap.                         |

---

## Training

### Training Process
1. **Data Splitting**:
   - The dataset is split into training, validation, and test sets.
2. **Loss Function**:
   - Mean Squared Error (MSE) loss is used to measure the difference between predicted and actual band gaps.
3. **Optimizer**:
   - Adam optimizer is used for training.
4. **Early Stopping**:
   - Training stops if the validation loss does not improve for a specified number of epochs (patience).

### Hyperparameters
| Parameter         | Value(s)                  | Description                                  |
|-------------------|---------------------------|----------------------------------------------|
| Batch Size        | 16, 32                    | Number of graphs per batch.                 |
| Learning Rate     | 0.001, 0.0005             | Step size for the optimizer.                |
| Epochs            | 50, 100                   | Number of training iterations.              |
| Hidden Dimension  | 128                       | Dimension of hidden layers.                 |
| Dropout           | 0.2                       | Dropout rate for regularization.            |
| Convolution Type  | `gcn`, `sage`             | Type of graph convolution layer.            |

---

## Evaluation

### Evaluation Metric
- **Root Mean Squared Error (RMSE)**:
  - Measures the difference between predicted and actual band gaps.
  - Lower RMSE indicates better performance.

### Evaluation Process
1. The model is evaluated on the validation set during training.
2. The best model (based on validation RMSE) is saved.
3. The test set is used to evaluate the final model.

---

## Final Output

The final output of the GNN is:
- **Predicted Band Gap**: A scalar value representing the band gap of the perovskite material.

---

## Summary of GNN Components

| Component            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Input Node Features  | Frequency-encoded values + one-hot encoding for site type.                 |
| Graph Convolution    | Updates node features by aggregating information from neighbors.           |
| Global Pooling       | Aggregates node features into a graph-level representation.                |
| MLP Head             | Maps the graph-level representation to the target property (band gap).     |
| Output               | Predicted band gap (scalar).                                               |

---

## How to Run

1. **Prepare the Input Data**:
   - Ensure the input CSV file is frequency-encoded and contains the required columns.

2. **Run the Script**:
   ```bash
   python GNN.py ../data/perovskite_frequency_encoded_by_site.csv
   ```

3. **Check Results**:
   - The trained model is saved in the `../models/` directory.
   - Training and evaluation logs are saved in `../results/results_log.csv`.

---

## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- Relevant research papers on GNNs for material science.

---

## License

This project is licensed under the MIT License.