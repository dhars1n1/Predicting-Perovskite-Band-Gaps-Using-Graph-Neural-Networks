import pandas as pd
import re
from math import isclose

DATA_PATH = "Data-Driven-Material-Modelling/data/perovskite_data.csv"

def clean_string(s):
    """Standardize string by stripping spaces and handling NaN."""
    if pd.isna(s):
        return ""
    return str(s).strip()

def parse_elements_and_coeffs(elem_str, coeff_str):
    """Parse multi-element strings with mixed separators and return dict."""
    elem_str = clean_string(elem_str)
    coeff_str = clean_string(coeff_str)

    if elem_str == "":
        return {}

    # Split elements by ';' or ','
    elements = re.split(r"[;,]", elem_str)
    elements = [e.strip() for e in elements if e.strip()]

    # Handle coefficients
    if coeff_str == "":
        coeffs = [1.0] * len(elements)
    else:
        coeffs = re.split(r"[;,]", coeff_str)
        coeffs = [c.strip() for c in coeffs if c.strip()]
        parsed_coeffs = []
        for c in coeffs:
            try:
                parsed_coeffs.append(float(c))
            except ValueError:
                match = re.search(r"[-+]?[0-9]*\.?[0-9]+", c)
                parsed_coeffs.append(float(match.group()) if match else 1.0)
        coeffs = parsed_coeffs

    # Pad/truncate if mismatch
    if len(coeffs) < len(elements):
        coeffs += [1.0] * (len(elements) - len(coeffs))
    if len(coeffs) > len(elements):
        coeffs = coeffs[:len(elements)]

    return dict(zip(elements, coeffs))

def parse_row(row):
    """Combine A, B, and C ions into one composition dict."""
    comp = {}
    # A site
    a_dict = parse_elements_and_coeffs(
        row.get("Perovskite_composition_a_ions", ""),
        row.get("Perovskite_composition_a_ions_coefficients", "")
    )
    # B site
    b_dict = parse_elements_and_coeffs(
        row.get("Perovskite_composition_b_ions", ""),
        row.get("Perovskite_composition_b_ions_coefficients", "")
    )
    # C site
    c_dict = parse_elements_and_coeffs(
        row.get("Perovskite_composition_c_ions", ""),
        row.get("Perovskite_composition_c_ions_coefficients", "")
    )

    # Merge all dicts
    for d in (a_dict, b_dict, c_dict):
        for el, val in d.items():
            comp[el] = comp.get(el, 0) + val

    return comp

def normalize_comp(comp):
    total = sum(comp.values())
    if isclose(total, 0.0):
        return comp
    return {el: val / total for el, val in comp.items()}

def clean_perovskite_data(path):
    df = pd.read_csv(path, low_memory=False)  # avoid dtype warning
    df.columns = [c.strip() for c in df.columns]
    df["parsed_composition"] = df.apply(parse_row, axis=1)
    df["composition_fraction"] = df["parsed_composition"].apply(normalize_comp)
    return df

if __name__ == "__main__":
    df_clean = clean_perovskite_data(DATA_PATH)
    df_clean.to_csv("perovskite_cleaned.csv", index=False)
    print(df_clean[["Ref_ID", "Perovskite_composition_short_form",
                    "parsed_composition", "composition_fraction"]].head())
