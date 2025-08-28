# Perovskite Materials Property Prediction using Graph Neural Networks

A robust chemistry-aware Graph Neural Network (GNN) implementation for predicting perovskite material properties using molecular composition and structure data.

## Overview

This project implements a Graph Isomorphism Network with Edge features (GINE) to predict perovskite material stability by learning from chemical composition, structural properties, and thermodynamic data. Each perovskite compound is represented as a molecular graph where atoms are nodes and chemical interactions are edges.

## Features

- **Chemistry-Aware Design**: Incorporates domain-specific knowledge including ionic radii, electronegativity, and perovskite site assignments (A-B-X structure)
- **Edge Features**: Sophisticated edge attributes capturing site-to-site interactions and chemical relationships
- **Robust Data Processing**: Handles multiple data formats, missing values, and various composition representations
- **Multi-Scale Learning**: Combines node-level (atomic), edge-level (bonding), and global-level (material) features
- **Production Ready**: Includes proper ML practices like feature normalization, early stopping, and gradient clipping

## Project Structure

```
├── GNN.py                      # Main implementation
├── requirements.txt            # Python dependencies
├── models/                     # Model checkpoints (created during training)
├── plots/
│   ├── best_checkpoint.pt      # Best model weights
│   ├── train_loss_final.png    # Training loss curves
│   └── val_metrics_final.png   # Validation metrics plots
└── ../data/
    └── perovskite_numeric_encoded.csv  # Input dataset
```

## Input Data Format

The model expects a CSV file with perovskite material data containing:

### Required Columns:
- `Perovskite_composition_a_ions`: A-site elements (e.g., "MA", "FA", "Cs")
- `Perovskite_composition_b_ions`: B-site elements (e.g., "Pb", "Sn")  
- `Perovskite_composition_c_ions`: X-site elements (e.g., "I", "Br", "Cl")
- `Perovskite_band_gap`: Band gap values (eV)
- `Stability_temperature_min`/`Stability_temperature_max`: Temperature stability range (K)

### Optional Columns:
- Coefficient columns for stoichiometry
- Parsed composition dictionaries
- Additional structural parameters

## Model Architecture

### Graph Representation:
- **Nodes**: Chemical elements with 9 features each:
  - Atomic number, mass, electronegativity
  - Ionic radius, composition fraction
  - Organic indicator, site assignment (A/B/X)

- **Edges**: Directed connections between all atom pairs with 17 features:
  - Site-to-site interaction type (9 one-hot categories)
  - Chemical property differences (ΔZ, Δelectronegativity, Δradius)
  - Geometric and compositional features

- **Global Features**: Material-level properties (4 features):
  - Band gap, average temperature
  - Tolerance factor, octahedral factor

### Network Architecture:
- 4-layer Graph Isomorphism Network with Edge features (GINEConv)
- 128 hidden dimensions
- Batch normalization and dropout for regularization
- Global pooling + MLP readout head

## Target Prediction

The model predicts a **proxy energy metric**:
```
E_proxy = bandgap × average_temperature
```
This serves as an indicator of material stability and performance potential.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the main dependencies:
```
torch
torch-geometric
pymatgen
pandas
numpy
matplotlib
```

## Usage

### Basic Training:
```bash
python GNN.py
```

### Configuration:
Edit the user settings section in `GNN.py`:
```python
INPUT_CSV = "../data/perovskite_numeric_encoded.csv"
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 80
RANDOM_SEED = 42
```

### Output:
The script will:
1. Load and preprocess the perovskite dataset
2. Build molecular graphs for each compound
3. Train the GNN model with validation monitoring
4. Save the best model checkpoint
5. Generate training/validation plots
6. Report final test set performance

## Results

The model outputs:
- **Training curves**: Loss progression over epochs
- **Validation metrics**: MAE and RMSE on validation set
- **Test performance**: Final model evaluation metrics
- **Model checkpoint**: Best performing model weights

## Key Features

### Robust Data Processing:
- Handles multiple composition formats (short form, long form, parsed dictionaries)
- NaN-safe operations throughout
- Flexible parsing for various delimiter formats

### Chemistry Integration:
- Uses pymatgen for accurate elemental properties
- Incorporates organic cation data (MA, FA, etc.)
- Site-specific ionic radii selection

### Advanced Training:
- Feature normalization across node, edge, and global levels
- Early stopping with patience
- Learning rate scheduling
- Gradient clipping for stability

## Performance

Typical performance metrics:
- **Training time**: ~10-20 minutes for 80 epochs
- **Memory usage**: Scales with dataset size and batch size
- **Accuracy**: Depends on data quality and target complexity

## Customization

### Adding New Features:
1. Modify the feature extraction functions in the "element property helpers" section
2. Update the `build_graph()` function to include new features
3. Adjust model input dimensions accordingly

### Different Targets:
Replace the target calculation in `build_graph()`:
```python
# Current: E_proxy = bandgap * tmean
# Example alternatives:
target = formation_energy  # Direct energy prediction
target = stability_score   # Stability classification
```

### Model Architecture:
Modify the `ChemGNN_Edge` class to:
- Change number of layers, hidden dimensions
- Add attention mechanisms
- Include different pooling strategies

## Troubleshooting

### Common Issues:
1. **Missing data**: Ensure required columns exist in CSV
2. **Memory errors**: Reduce batch size or use gradient accumulation  
3. **NaN losses**: Check for invalid target values or extreme feature ranges
4. **Poor convergence**: Adjust learning rate, add more regularization

### Data Requirements:
- Minimum: ~100 samples for meaningful training
- Recommended: 1000+ samples for robust performance
- Each sample needs valid bandgap and temperature data

## Citation

If you use this code, please cite the relevant papers on:
- Graph Isomorphism Networks (Xu et al.)
- Materials property prediction using GNNs
- Your specific perovskite dataset source