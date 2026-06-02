"""
Process and clean TCGA-HNSC clinical data for the dashboard.

Takes the raw CSV from download_data.py and produces a clean,
analysis-ready dataset with derived features.

Usage:
    python process_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH = Path("data/tcga_hnsc_raw.csv")
OUTPUT_PATH = Path("data/tcga_hnsc_clean.csv")


def clean_stage(stage_str):
    """Normalize AJCC stage strings to broad categories."""
    if pd.isna(stage_str) or stage_str in ("not reported", "Not Reported"):
        return "Not Reported"
    stage = stage_str.upper().replace("STAGE ", "").strip()
    if stage.startswith("IV"):
        return "Stage IV"
    elif stage.startswith("III"):
        return "Stage III"
    elif stage.startswith("II"):
        return "Stage II"
    elif stage.startswith("I") or stage == "0":
        return "Stage I"
    return "Not Reported"


def clean_anatomic_site(site_str):
    """Group anatomic sites into clinically meaningful categories."""
    if pd.isna(site_str) or site_str in ("not reported", "Not Reported", "Unknown"):
        return "Other/Unknown"

    site = site_str.lower()

    if any(k in site for k in ["base of tongue"]):
        return "Oropharynx"
    elif any(k in site for k in ["tongue", "lingual", "ventral surface of tongue"]):
        return "Tongue"
    elif any(k in site for k in ["larynx", "glottis", "supraglott", "subglott"]):
        return "Larynx"
    elif any(k in site for k in ["tonsil", "oropharynx", "oropharyngeal", "vallecula", "pharynx, nos"]):
        return "Oropharynx"
    elif any(k in site for k in ["floor of mouth", "anterior floor"]):
        return "Floor of Mouth"
    elif any(k in site for k in ["gum", "gingiv", "palate", "buccal", "oral", "lip",
                                   "mouth", "cheek", "retromolar", "mandible", "upper gum",
                                   "overlapping lesion of lip"]):
        return "Oral Cavity"
    elif any(k in site for k in ["hypopharynx", "pyriform", "postcricoid"]):
        return "Hypopharynx"
    elif any(k in site for k in ["sinus", "nasal", "nasopharyn"]):
        return "Nasal/Sinus"
    return "Other/Unknown"


def derive_hpv_proxy(row):
    """
    Derive HPV status proxy from anatomic site.

    Oropharyngeal cancers (tonsil, base of tongue, oropharynx NOS)
    are strongly associated with HPV. This is a site-based proxy,
    not molecular HPV testing.

    References:
    - Gillison et al., JAMA 2000: ~60-70% of oropharyngeal SCC are HPV+
    - Chaturvedi et al., JCO 2011: rising HPV+ proportion in oropharynx
    """
    if row.get("anatomic_site_group") == "Oropharynx":
        return "Likely HPV-associated"
    elif row.get("anatomic_site_group") in ("Other/Unknown",):
        return "Unknown"
    else:
        return "Likely HPV-negative"


def derive_survival_time(row):
    """
    Compute overall survival time in days.
    Use days_to_death if dead, days_to_last_follow_up if alive (censored).
    """
    if row["vital_status"] == "Dead":
        for col in ["days_to_death_dx", "days_to_death_demo"]:
            if pd.notna(row.get(col)):
                return row[col]
    if pd.notna(row.get("days_to_last_follow_up")):
        return row["days_to_last_follow_up"]
    return np.nan


def build_treatment_strategy(row):
    """Create a readable treatment strategy label."""
    parts = []
    if row.get("had_surgery"):
        parts.append("Surgery")
    if row.get("had_radiation"):
        parts.append("Radiation")
    if row.get("had_pharmaceutical"):
        parts.append("Chemo/Pharma")
    if not parts:
        return "No treatment recorded"
    return " + ".join(parts)


def main():
    print(f"Reading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows.")

    # Clean and derive columns
    df["stage_group"] = df["ajcc_pathologic_stage"].apply(clean_stage)

    # Use tissue_or_organ for anatomic site (not site_of_resection which is too generic)
    df["anatomic_site_group"] = df["tissue_or_organ"].apply(clean_anatomic_site)

    # HPV proxy based on anatomic site
    df["hpv_proxy"] = df.apply(derive_hpv_proxy, axis=1)

    # Age: GDC reports age_at_diagnosis in days; convert to years
    if df["age_at_diagnosis_days"].max() > 200:
        df["age_at_diagnosis_years"] = (df["age_at_diagnosis_days"] / 365.25).round(1)
    else:
        df["age_at_diagnosis_years"] = df["age_at_diagnosis_days"]

    # Age bins
    bins = [0, 45, 55, 65, 75, 200]
    labels = ["<45", "45-54", "55-64", "65-74", "75+"]
    df["age_group"] = pd.cut(df["age_at_diagnosis_years"], bins=bins, labels=labels, right=False)

    # Survival
    df["survival_days"] = df.apply(derive_survival_time, axis=1)
    df["survival_months"] = (df["survival_days"] / 30.44).round(1)
    df["event_observed"] = (df["vital_status"] == "Dead").astype(int)

    # Treatment strategy
    df["treatment_strategy"] = df.apply(build_treatment_strategy, axis=1)

    # Clean tobacco
    df["smoking_status"] = df["tobacco_smoking_status"].replace({
        "Not Reported": "Unknown",
        "not reported": "Unknown",
        np.nan: "Unknown",
    })

    # Clean alcohol
    df["alcohol_status"] = df["alcohol_history"].replace({
        "Not Reported": "Unknown",
        "not reported": "Unknown",
        np.nan: "Unknown",
    })

    # T staging simplified
    df["t_stage"] = df["ajcc_pathologic_t"].fillna("Unknown").str.upper()
    df["n_stage"] = df["ajcc_pathologic_n"].fillna("Unknown").str.upper()

    # Select columns for dashboard
    keep_cols = [
        "case_id",
        "gender",
        "race",
        "age_at_diagnosis_years",
        "age_group",
        "primary_diagnosis",
        "stage_group",
        "t_stage",
        "n_stage",
        "anatomic_site_group",
        "hpv_proxy",
        "treatment_strategy",
        "had_radiation",
        "had_pharmaceutical",
        "had_surgery",
        "smoking_status",
        "alcohol_status",
        "pack_years_smoked",
        "vital_status",
        "survival_days",
        "survival_months",
        "event_observed",
    ]

    df_clean = df[[c for c in keep_cols if c in df.columns]].copy()

    # Drop rows with no survival data (can't analyze them)
    n_before = len(df_clean)
    df_clean = df_clean.dropna(subset=["survival_days"])
    print(f"Dropped {n_before - len(df_clean)} rows with no survival data.")

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved clean data to {OUTPUT_PATH} ({len(df_clean)} rows, {len(df_clean.columns)} columns)")

    # Summary statistics
    print(f"\n--- Data Summary ---")
    print(f"Patients: {len(df_clean)}")
    print(f"Events (deaths): {df_clean['event_observed'].sum()}")
    print(f"Median survival: {df_clean['survival_months'].median():.1f} months")
    print(f"\nStage distribution:\n{df_clean['stage_group'].value_counts()}")
    print(f"\nSite distribution:\n{df_clean['anatomic_site_group'].value_counts()}")
    print(f"\nHPV proxy:\n{df_clean['hpv_proxy'].value_counts()}")
    print(f"\nTreatment strategies:\n{df_clean['treatment_strategy'].value_counts()}")


if __name__ == "__main__":
    main()
