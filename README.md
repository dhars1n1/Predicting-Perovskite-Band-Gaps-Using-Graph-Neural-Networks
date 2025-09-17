# Data-Driven Material Modeling (DDMM)

This repository contains code and resources for predicting material properties using a Graph Neural Network (GNN) approach. The project focuses on perovskite materials, leveraging their structural and compositional data to predict properties like band gaps. The pipeline includes data preprocessing, GNN training, and selecting the best perovskites for specific applications.

---

## Getting Started

### 1. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies. Run the following commands to set up the environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# For Windows, use: venv\Scripts\activate
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Project Overview

This project is divided into three main stages:

1. **Data Preprocessing**: Cleaning and transforming raw perovskite data into graph-compatible formats.
2. **GNN Training**: Training a Graph Neural Network to predict material properties.
3. **Best Perovskite Selection**: Using the trained model to predict and rank perovskites based on their suitability for specific applications.

---

## 1. Data Preprocessing

The preprocessing pipeline is implemented in `data_preprocessing.py`. It involves the following steps:

### Step 1: Basic Cleaning
- Load raw data from `data/perovskite_data.csv`.
- Select relevant columns, remove duplicates, and handle missing values.
- Save the cleaned data as `data/perovskite_filtered.csv`.

### Step 2: Extract Unique Species
- Extract unique atomic species from the composition columns.
- Save the list of unique species as `data/unique_species.csv`.

### Step 3: Frequency and Categorization
- Categorize atomic species into **A-site**, **B-site**, **X-site**, or **Additive/Other** based on predefined sets.
- Generate a summary of species frequencies and categories, saved as `data/species_summary.csv`.

### Step 4: Frequency Encoding (Global)
- Perform frequency encoding for all species across A, B, and C sites.
- Save the frequency-encoded dataset as `data/perovskite_frequency_encoded.csv`.

### Step 5: Frequency Encoding by Site
- Perform frequency encoding separately for A, B, and C sites.
- Save the site-separated frequency-encoded dataset as `data/perovskite_frequency_encoded_by_site.csv`.

---

## 2. GNN Training

The GNN training pipeline is implemented in `GNN.py`. It involves the following steps:

### Graph Construction
- Each perovskite is represented as a graph:
  - **Nodes**: Represent A-site, B-site, and C-site features.
  - **Edges**: Represent relationships between the sites.
- Node features include frequency-encoded values and one-hot encodings for site types.

### GNN Architecture
The GNN model is implemented using PyTorch Geometric and consists of:
1. **Graph Convolution Layers**:
   - Aggregates information from neighboring nodes using either `GCNConv` or `SAGEConv`.
2. **Global Pooling Layer**:
   - Pools node features into a graph-level representation using global mean pooling.
3. **MLP (Multi-Layer Perceptron)**:
   - Maps the graph representation to the target property (e.g., band gap).

The architecture is defined as:
```python
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
```

### Training and Evaluation
- The dataset is split into training, validation, and test sets.
- The model is trained using Mean Squared Error (MSE) loss.
- Hyperparameters such as batch size, learning rate, number of epochs, and convolution type are tuned using grid search.
- Early stopping is used to prevent overfitting.

### Outputs
- The best model is saved in the `models/` directory.
- Training and evaluation results are logged in `results/results_log.csv`.

---

## 3. Best Perovskite Selection

The best perovskites are selected using the trained GNN model in `pick_best_perovskite.py`. The process involves:

### Step 1: Load the Trained Model
- The trained GNN model and scaler are loaded from the saved checkpoint.

### Step 2: Convert Rows to Graphs
- Each row in the frequency-encoded dataset is converted into a graph compatible with the GNN model.

### Step 3: Predict Band Gaps
- The GNN model predicts the band gap for each perovskite.
- Predictions are scaled back to the original range using the scaler.

### Step 4: Rank Perovskites
- Perovskites are ranked based on their predicted band gaps.
- The top candidates for solar cell applications are identified.

### Outputs
- Predictions are saved in `results/predicted_perovskites.csv`.
- The top 5 candidates are displayed in the terminal.

---

## Example Usage

### 1. Data Preprocessing
Run the preprocessing script to clean and encode the data:
```bash
python data_preprocessing.py
```

### 2. Train the GNN
Train the GNN model using the preprocessed data:
```bash
python GNN.py
```

### 3. Predict and Rank Perovskites
Use the trained model to predict and rank perovskites:
```bash
python pick_best_perovskite.py
```

---

## Directory Structure

```
DDMM/
├── data/                     # Raw and processed data
├── models/                   # Saved GNN models
├── results/                  # Logs and predictions
├── data_preprocessing.py     # Data preprocessing script
├── GNN.py                    # GNN training script
├── pick_best_perovskite.py   # Best perovskite selection script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- Relevant research papers on GNNs for material science.

---

## License

This project is licensed under the MIT License.