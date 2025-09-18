import pandas as pd
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

print("Loading raw data.")
df = pd.read_csv("perovskite_data.csv")

# --- Step 1: Basic Cleaning ---
print("Selecting and cleaning columns.")
columns_to_keep = [
    "Perovskite_composition_a_ions",
    "Perovskite_composition_a_ions_coefficients",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_b_ions_coefficients",
    "Perovskite_composition_c_ions",
    "Perovskite_composition_c_ions_coefficients",
    "Perovskite_band_gap"
]
df = df[columns_to_keep]
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df = df.dropna(how="any")
df = df[(df != "").all(axis=1)]
df = df.drop_duplicates()
df.to_csv("perovskite_filtered.csv", index=False)
print(f"Filtered data saved as perovskite_filtered.csv ({len(df)} rows)")

# --- Step 2: Extract Unique Species ---
def extract_species(text):
    if pd.isna(text):
        return []
    # Normalize: uppercase, strip
    tokens = [t.strip().upper() for t in re.split(r"[^A-Za-z0-9]+", str(text)) if t and t.lower() != "nan" and not t.isnumeric()]
    return tokens

ion_columns = [
    "Perovskite_composition_a_ions",
    "Perovskite_composition_b_ions",
    "Perovskite_composition_c_ions"
]

all_species = set()
for col in ion_columns:
    df[col].dropna().apply(lambda x: all_species.update(extract_species(x)))
unique_species = sorted(all_species)
pd.DataFrame(unique_species, columns=["Species"]).to_csv("data/unique_species.csv", index=False)
print(f"Found {len(unique_species)} unique species. Saved as unique_species.csv")

# --- Step 3: Frequency & Categorization ---
A_SITE = {"CS", "FA", "MA", "RB", "K", "NA", "LI", "BA", "CA", "SR", "EA", "PEA", "DMA", "NMA", "GU", "AVA", "BDA", "EDA", "HEA", "IPA", "TBA"}
B_SITE = {"PB", "SN", "GE", "BI", "SB", "TI", "ZR", "NB", "MN", "FE", "CO", "NI", "CU", "ZN", "MG"}
X_SITE = {"I", "BR", "CL", "F", "SCN", "PF6"}

def categorize(species):
    if species in A_SITE:
        return "A-site"
    elif species in B_SITE:
        return "B-site"
    elif species in X_SITE:
        return "X-site"
    else:
        return "Additive/Other"

species_data = {col: [] for col in ion_columns}
for col in ion_columns:
    df[col].dropna().apply(lambda x: species_data[col].extend(extract_species(x)))
all_species_flat = sum(species_data.values(), [])
freq_counter = Counter(all_species_flat)
species_summary = pd.DataFrame(
    [(sp, freq, categorize(sp)) for sp, freq in freq_counter.items()],
    columns=["Species", "Frequency", "Category"]
).sort_values(by="Frequency", ascending=False)
species_summary.to_csv("species_summary.csv", index=False)
print(f"Species summary with categories saved as species_summary.csv")

# --- Step 4: Frequency Encoding (Global) ---
print("Frequency encoding (global, all sites together).")
species_cols = {
    "Perovskite_composition_a_ions": "Perovskite_composition_a_ions_coefficients",
    "Perovskite_composition_b_ions": "Perovskite_composition_b_ions_coefficients",
    "Perovskite_composition_c_ions": "Perovskite_composition_c_ions_coefficients"
}

all_species = sorted([s for s in set(all_species_flat) if s])
freq_encoded = pd.DataFrame(0, index=df.index, columns=all_species)

def extract_coeffs(text):
    if pd.isna(text):
        return []
    return [c.strip() for c in re.split(r"[^A-Za-z0-9.\-eE]+", str(text)) if c and c.lower() != "nan"]

for ion_col, coeff_col in species_cols.items():
    for i, row in df.iterrows():
        species_list = extract_species(row[ion_col])
        coeff_list = extract_coeffs(row[coeff_col])
        for sp, coeff in zip(species_list, coeff_list):
            if sp in freq_encoded.columns:
                try:
                    freq_encoded.at[i, sp] += float(coeff)
                except Exception:
                    freq_encoded.at[i, sp] += 1  # fallback if coeff is non-numeric

freq_encoded["Perovskite_band_gap"] = df["Perovskite_band_gap"].values
freq_encoded = freq_encoded.loc[:, (freq_encoded != 0).any(axis=0)]  # drop all-zero columns
freq_encoded.to_csv("perovskite_frequency_encoded.csv", index=False)
print(f"Frequency encoded dataset saved as perovskite_frequency_encoded.csv ({freq_encoded.shape[1]-1} features)")

# --- Step 5: Frequency Encoding by Site (A/B/C) ---
print("Frequency encoding by site (A/B/C).")
encoded_blocks = []

for site, (ion_col, coeff_col) in zip(["A", "B", "C"], species_cols.items()):
    site_species = set()
    df[ion_col].dropna().apply(lambda x: site_species.update(extract_species(x)))
    site_species = sorted([s for s in site_species if s])

    # Initialize with float dtype to avoid FutureWarning
    site_encoded = pd.DataFrame(0.0, index=df.index, columns=[f"{site}_{s}" for s in site_species])

    for i, row in df.iterrows():
        species_list = extract_species(row[ion_col])
        coeff_list = extract_coeffs(row[coeff_col])
        for sp, coeff in zip(species_list, coeff_list):
            col_name = f"{site}_{sp}"
            if col_name in site_encoded.columns:
                try:
                    site_encoded.at[i, col_name] += float(coeff)
                except Exception:
                    site_encoded.at[i, col_name] += 1.0   # fallback, keep float

    # Drop columns that are all zero
    site_encoded = site_encoded.loc[:, (site_encoded != 0).any(axis=0)]
    encoded_blocks.append(site_encoded)

# Merge all blocks + band gap
freq_encoded_by_site = pd.concat(encoded_blocks, axis=1)
freq_encoded_by_site["Perovskite_band_gap"] = df["Perovskite_band_gap"].values

freq_encoded_by_site.to_csv("perovskite_frequency_encoded_by_site.csv", index=False)
print(f"Frequency encoding with A/B/C site separation saved as perovskite_frequency_encoded_by_site.csv ({freq_encoded_by_site.shape[1]-1} features)")
