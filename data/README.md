# Data Preprocessing for Perovskite Band Gap Prediction

This document provides a detailed explanation of the data preprocessing steps used in the pipeline for predicting the band gap of perovskite materials. The preprocessing ensures that the raw data is cleaned, transformed, and prepared for use in the Graph Neural Network (GNN) model.

---

## Overview of Data Preprocessing

The preprocessing pipeline is designed to:
1. Clean and filter the raw data.
2. Extract unique atomic species from the dataset.
3. Categorize atomic species into specific sites (A-site, B-site, C-site).
4. Perform frequency encoding of atomic species to create numerical features.
5. Prepare the data in a graph-compatible format for the GNN model.

Each step in the pipeline is critical to ensure the data is consistent, complete, and suitable for graph-based learning.

---

## Preprocessing Steps

### Step 1: Basic Cleaning
- **Purpose**: Remove irrelevant or incomplete data to ensure the dataset is clean and reliable.
- **Operations**:
  - Load raw data from `data/perovskite_data.csv`.
  - Select relevant columns required for the analysis.
  - Remove duplicate rows to avoid redundancy.
  - Handle missing values by filtering out rows with incomplete data.
- **Output**: A cleaned dataset saved as `data/perovskite_filtered.csv`.

### Step 2: Extract Unique Atomic Species
- **Purpose**: Identify all unique atomic species present in the dataset for further categorization and encoding.
- **Operations**:
  - Extract unique atomic species from the composition columns.
  - Save the list of unique species as `data/unique_species.csv`.
- **Why**: This step ensures that all atomic species are accounted for and can be categorized appropriately.

### Step 3: Frequency and Categorization
- **Purpose**: Categorize atomic species into specific sites (A-site, B-site, C-site) based on their roles in the perovskite structure.
- **Operations**:
  - Categorize atomic species into:
    - **A-site**: Typically large cations.
    - **B-site**: Smaller cations.
    - **C-site**: Anions or other species.
    - **Additive/Other**: Species that do not fit into the above categories.
  - Generate a summary of species frequencies and categories.
  - Save the summary as `data/species_summary.csv`.
- **Why**: Categorization is essential for creating meaningful node features in the graph representation.

### Step 4: Frequency Encoding (Global)
- **Purpose**: Convert atomic species into numerical features using frequency encoding.
- **Operations**:
  - Calculate the frequency of each atomic species across all sites.
  - Assign a numerical value to each species based on its frequency.
  - Save the frequency-encoded dataset as `data/perovskite_frequency_encoded.csv`.
- **Why**: Frequency encoding provides a numerical representation of atomic species, which is required for machine learning models.

### Step 5: Frequency Encoding by Site
- **Purpose**: Perform frequency encoding separately for A-site, B-site, and C-site to capture site-specific information.
- **Operations**:
  - Calculate the frequency of each atomic species within each site (A, B, C).
  - Assign numerical values to species based on their site-specific frequencies.
  - Save the site-separated frequency-encoded dataset as `data/perovskite_frequency_encoded_by_site.csv`.
- **Why**: Site-specific encoding ensures that the model can differentiate between the roles of atomic species in different sites.

---

## Preprocessing Pipeline

The preprocessing pipeline can be summarized as follows:

1. **Load Raw Data**:
   - Load the raw dataset from `data/perovskite_data.csv`.
2. **Clean Data**:
   - Remove duplicates and handle missing values.
3. **Extract Unique Species**:
   - Identify all unique atomic species in the dataset.
4. **Categorize Species**:
   - Categorize species into A-site, B-site, C-site, or Additive/Other.
5. **Frequency Encoding**:
   - Perform global frequency encoding for all species.
   - Perform site-specific frequency encoding for A-site, B-site, and C-site.
6. **Save Processed Data**:
   - Save the cleaned and encoded datasets for use in the GNN model.

---

## Input Dimensions

### Node Feature Dimensions
Each node in the graph represents an atomic site (A-site, B-site, or C-site) and has the following features:
1. **Frequency-Encoded Values**:
   - Numerical representation of the atomic species based on their frequency.
2. **One-Hot Encoding**:
   - A binary vector indicating the site type (A-site, B-site, or C-site).

### Graph Input Dimensions
- **Node Feature Dimension**: The concatenated vector of frequency-encoded values and one-hot encoding.
- **Graph Input Dimension**: The combined feature vectors of all nodes in the graph.

---

## Why These Steps Are Necessary

1. **Cleaning**: Ensures the dataset is free from inconsistencies and missing values.
2. **Categorization**: Helps differentiate the roles of atomic species in the perovskite structure.
3. **Frequency Encoding**: Converts categorical data into numerical features suitable for machine learning.
4. **Site-Specific Encoding**: Captures the unique roles of atomic species in different sites, improving the model's ability to learn meaningful patterns.

---

## Summary

The preprocessing pipeline transforms raw perovskite data into a graph-compatible format by:
- Cleaning and filtering the data.
- Categorizing atomic species into specific sites.
- Encoding atomic species into numerical features.
- Preparing the data for use in the GNN model.

This ensures that the data is consistent, complete, and ready for graph-based learning.

```# Data Preprocessing for Perovskite Band Gap Prediction

This document provides a detailed explanation of the data preprocessing steps used in the pipeline for predicting the band gap of perovskite materials. The preprocessing ensures that the raw data is cleaned, transformed, and prepared for use in the Graph Neural Network (GNN) model.

---

## Overview of Data Preprocessing

The preprocessing pipeline is designed to:
1. Clean and filter the raw data.
2. Extract unique atomic species from the dataset.
3. Categorize atomic species into specific sites (A-site, B-site, C-site).
4. Perform frequency encoding of atomic species to create numerical features.
5. Prepare the data in a graph-compatible format for the GNN model.

Each step in the pipeline is critical to ensure the data is consistent, complete, and suitable for graph-based learning.

---

## Preprocessing Steps

### Step 1: Basic Cleaning
- **Purpose**: Remove irrelevant or incomplete data to ensure the dataset is clean and reliable.
- **Operations**:
  - Load raw data from `data/perovskite_data.csv`.
  - Select relevant columns required for the analysis.
  - Remove duplicate rows to avoid redundancy.
  - Handle missing values by filtering out rows with incomplete data.
- **Output**: A cleaned dataset saved as `data/perovskite_filtered.csv`.

### Step 2: Extract Unique Atomic Species
- **Purpose**: Identify all unique atomic species present in the dataset for further categorization and encoding.
- **Operations**:
  - Extract unique atomic species from the composition columns.
  - Save the list of unique species as `data/unique_species.csv`.
- **Why**: This step ensures that all atomic species are accounted for and can be categorized appropriately.

### Step 3: Frequency and Categorization
- **Purpose**: Categorize atomic species into specific sites (A-site, B-site, C-site) based on their roles in the perovskite structure.
- **Operations**:
  - Categorize atomic species into:
    - **A-site**: Typically large cations.
    - **B-site**: Smaller cations.
    - **C-site**: Anions or other species.
    - **Additive/Other**: Species that do not fit into the above categories.
  - Generate a summary of species frequencies and categories.
  - Save the summary as `data/species_summary.csv`.
- **Why**: Categorization is essential for creating meaningful node features in the graph representation.

### Step 4: Frequency Encoding (Global)
- **Purpose**: Convert atomic species into numerical features using frequency encoding.
- **Operations**:
  - Calculate the frequency of each atomic species across all sites.
  - Assign a numerical value to each species based on its frequency.
  - Save the frequency-encoded dataset as `data/perovskite_frequency_encoded.csv`.
- **Why**: Frequency encoding provides a numerical representation of atomic species, which is required for machine learning models.

### Step 5: Frequency Encoding by Site
- **Purpose**: Perform frequency encoding separately for A-site, B-site, and C-site to capture site-specific information.
- **Operations**:
  - Calculate the frequency of each atomic species within each site (A, B, C).
  - Assign numerical values to species based on their site-specific frequencies.
  - Save the site-separated frequency-encoded dataset as `data/perovskite_frequency_encoded_by_site.csv`.
- **Why**: Site-specific encoding ensures that the model can differentiate between the roles of atomic species in different sites.

---

## Preprocessing Pipeline

The preprocessing pipeline can be summarized as follows:

1. **Load Raw Data**:
   - Load the raw dataset from `data/perovskite_data.csv`.
2. **Clean Data**:
   - Remove duplicates and handle missing values.
3. **Extract Unique Species**:
   - Identify all unique atomic species in the dataset.
4. **Categorize Species**:
   - Categorize species into A-site, B-site, C-site, or Additive/Other.
5. **Frequency Encoding**:
   - Perform global frequency encoding for all species.
   - Perform site-specific frequency encoding for A-site, B-site, and C-site.
6. **Save Processed Data**:
   - Save the cleaned and encoded datasets for use in the GNN model.

---

## Input Dimensions

### Node Feature Dimensions
Each node in the graph represents an atomic site (A-site, B-site, or C-site) and has the following features:
1. **Frequency-Encoded Values**:
   - Numerical representation of the atomic species based on their frequency.
2. **One-Hot Encoding**:
   - A binary vector indicating the site type (A-site, B-site, or C-site).

### Graph Input Dimensions
- **Node Feature Dimension**: The concatenated vector of frequency-encoded values and one-hot encoding.
- **Graph Input Dimension**: The combined feature vectors of all nodes in the graph.

---

## Why These Steps Are Necessary

1. **Cleaning**: Ensures the dataset is free from inconsistencies and missing values.
2. **Categorization**: Helps differentiate the roles of atomic species in the perovskite structure.
3. **Frequency Encoding**: Converts categorical data into numerical features suitable for machine learning.
4. **Site-Specific Encoding**: Captures the unique roles of atomic species in different sites, improving the model's ability to learn meaningful patterns.

---

## Summary

The preprocessing pipeline transforms raw perovskite data into a graph-compatible format by:
- Cleaning and filtering the data.
- Categorizing atomic species into specific sites.
- Encoding atomic species into numerical features.
- Preparing the data for use in the GNN model.

This ensures that the data is consistent, complete, and ready for graph-based learning.
