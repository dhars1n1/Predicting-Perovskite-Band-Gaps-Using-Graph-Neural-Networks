# SHAP Analysis for Perovskite Band Gap Prediction

This folder provides scripts, notebooks, and documentation for performing SHAP (SHapley Additive exPlanations) analysis on the models developed in the [Predicting-Perovskite-Band-Gaps-Using-Graph-Neural-Networks](https://github.com/dhars1n1/Predicting-Perovskite-Band-Gaps-Using-Graph-Neural-Networks) project. SHAP analysis helps interpret the predictions of machine learning models by quantifying the contribution of each input feature.

---

## Table of Contents

- [Background](#background)
- [Folder Structure](#folder-structure)
- [Setup & Dependencies](#setup--dependencies)
- [Usage](#usage)
  - [Running SHAP Analysis](#running-shap-analysis)
  - [Visualizing SHAP Results](#visualizing-shap-results)
- [Sample Workflow](#sample-workflow)
- [Output & Interpretation](#output--interpretation)


---

## Background

Perovskites are a class of materials with wide-ranging applications in photovoltaics and optoelectronics. Predicting their electronic band gaps is crucial for designing new materials. This project employs Graph Neural Networks (GNNs) to predict band gaps based on compositional and structural features.

However, GNNs and other complex models are typically black boxes. SHAP provides a unified framework to interpret these models by assigning an importance value to each feature for individual predictions and globally across the dataset.

---

## Folder Structure

```
SHAP-Analysis/
├── shap.py
├── shap_correlation_analysis.png
├── shap_correlation_top.png
├── shap_site_analysis.png
├── shap_site_analysis_top.png
├── shap_summary_plot.png
```

- **scripts/**: Main code for running SHAP analysis (e.g., `run_shap.py`, `plot_shap.py`).
- **notebooks/**: Notebooks demonstrating step-by-step usage and interpretation.
- **results/**: Generated SHAP summary plots, dependence plots, and exported SHAP values.
- **example_data/**: Sample models and datasets for quick testing.

---

## Setup & Dependencies

To use the scripts in this folder, install the following Python packages:

```bash
pip install shap numpy pandas matplotlib scikit-learn torch
```

- **shap**: For SHAP value computation and visualization.
- **numpy/pandas**: For data manipulation.
- **matplotlib**: Plotting SHAP values.
- **scikit-learn**: Utilities for preprocessing and model interfaces.
- **torch**: For PyTorch-based GNN models.

You may need additional packages depending on your model architecture (e.g., DGL, PyTorch Geometric).

---

## Usage

### Running SHAP Analysis

1. **Prepare Your Model and Data**  
   Train your GNN or other ML model using your data. Save the trained model in a compatible format (e.g., PyTorch `.pt` file).

2. **Load Model and Data**  
   Use the scripts or notebooks in this folder to load your trained model and the input data for SHAP analysis.

3. **Initialize the SHAP Explainer**  
   Depending on your model type (tree, neural network, custom), initialize the appropriate SHAP explainer:
   ```python
   import shap
   # For a PyTorch model:
   explainer = shap.DeepExplainer(model, background_data)
   # For a scikit-learn model:
   explainer = shap.TreeExplainer(model)
   ```

4. **Compute SHAP Values**
   ```python
   shap_values = explainer.shap_values(test_data)
   ```

### Visualizing SHAP Results

- **Summary Plot**
  ```python
  shap.summary_plot(shap_values, test_data, feature_names=feature_names)
  ```
- **Dependence Plot**
  ```python
  shap.dependence_plot("feature_1", shap_values, test_data)
  ```
- **Force Plot (Individual Prediction)**
  ```python
  shap.force_plot(explainer.expected_value, shap_values[0,:], test_data.iloc[0,:])
  ```

---

## Sample Workflow

1. **Train your model** (see main repo for training scripts).
2. **Export your test dataset** (CSV, NumPy, or DataFrame).
3. **Run `run_shap.py`** to calculate SHAP values:
   ```bash
   python scripts/run_shap.py --model path/to/model.pt --data path/to/test_data.csv --output results/shap_values.csv
   ```
4. **Visualize using `shap.py`**:
   

Alternatively, open and run the Jupyter notebooks in the `notebooks/` folder for interactive analysis.

---

## Output & Interpretation

- **SHAP Summary Plot**: Shows global feature importance. Features at the top are most influential in predicting band gaps.
- **Dependence Plot**: Shows how individual features affect the prediction.
- **Force Plot**: Explains a single prediction by showing positive/negative contributions of each feature.
- **Exported CSVs**: SHAP values for each sample and feature for further statistical analysis.

Use these plots and tables to:
- Identify which atomic, compositional, or structural features most influence band gap predictions.
- Validate scientific hypotheses about perovskite properties.
- Debug and improve your model.

---
