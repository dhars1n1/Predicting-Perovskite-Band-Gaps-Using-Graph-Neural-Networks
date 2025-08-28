# Data-Driven Material Modelling

This repository contains code and resources for data-driven material modeling using a Graph Neural Network (GNN) approach. The goal is to predict material properties by learning directly from structural and compositional data represented as graphs.

## Overview

Material modeling is essential for understanding the behavior and properties of materials based on their composition and structure. Traditional methods often require significant domain expertise and manual feature engineering. This project leverages GNNs to automate feature extraction and prediction by treating materials as graphs, where nodes and edges represent atoms and bonds respectively.

## GNN Implementation

The core of this project is a Graph Neural Network, implemented using PyTorch Geometric (`torch_geometric`) for efficient graph data handling. The GNN architecture typically consists of:

- **Node Embedding Layer:** Initializes node features based on atomic properties (e.g., atom type, electronegativity).
- **Graph Convolution Layers:** Aggregates information from neighboring nodes (atoms/bonds) using multiple GNN layers (e.g., GCNConv, GraphSAGE, or Message Passing).
- **Readout Layer:** Pools node features to generate a graph-level representation (e.g., global mean pooling or attention pooling).
- **Output Layer:** Maps the graph representation to the target material property (regression or classification).

The model is trained end-to-end using supervised learning, optimizing for prediction accuracy on material properties.

## Input Data

- **Graphs:** Each material is represented as a graph, where:
  - **Nodes** represent atoms, with features such as element type, charge, or other atomic descriptors.
  - **Edges** represent bonds or relationships between atoms, possibly including bond type or distance.
- **Node Features:** Vector representations of atomic properties.
- **Edge Features:** (Optional) Bond characteristics, such as bond order or length.
- **Target Property:** The property to predict (e.g., Young's modulus, band gap, conductivity).

The input format typically follows the conventions of PyTorch Geometric:
```python
from torch_geometric.data import Data

data = Data(x=node_features, edge_index=edge_indices, edge_attr=edge_features, y=target)
```

## Output

- **Material Property Prediction:** For each input graph (material), the model outputs a prediction for the specified property (e.g., scalar value for regression tasks).
- **Batch Output:** When processing batches, outputs are vectors of predictions corresponding to each material in the batch.
- **Evaluation Metrics:** Common metrics include Mean Squared Error (MSE) for regression and accuracy for classification.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- Other dependencies as listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/dhars1n1/Data-Driven-Material-Modelling.git
cd Data-Driven-Material-Modelling
pip install -r requirements.txt
```

### Usage

1. **Prepare Data:** Format your material data as graphs compatible with PyTorch Geometric.
2. **Configure Model:** Adjust GNN architecture in the source code as needed.
3. **Train Model:** Run training scripts to fit the GNN to your data.
4. **Predict:** Use the trained model to predict properties of new materials.

Example usage:
```python
from model import MaterialGNN
model = MaterialGNN(...)
output = model(data)
```

## Directory Structure


## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- Relevant research papers on GNNs for material science

## License

This project is licensed under the MIT License.

---

For questions or contributions, please open an issue or pull request.