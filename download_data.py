"""
Download TCGA-HNSC clinical data from the GDC API.

This script pulls clinical, demographic, exposure, and treatment data
for all Head and Neck Squamous Cell Carcinoma (HNSC) cases from
The Cancer Genome Atlas (TCGA) via the Genomic Data Commons (GDC) API.

No authentication required — all clinical metadata is open access.

Usage:
    python download_data.py
"""

import requests
import json
import pandas as pd
from pathlib import Path

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
PROJECT_ID = "TCGA-HNSC"
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fields to pull from GDC
FIELDS = [
    "submitter_id",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.year_of_birth",
    "demographic.age_at_index",
    "diagnoses.age_at_diagnosis",
    "diagnoses.primary_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.site_of_resection_or_biopsy",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.morphology",
    "diagnoses.treatments.treatment_type",
    "diagnoses.treatments.treatment_or_therapy",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_death",
    "diagnoses.vital_status",
    "exposures.alcohol_history",
    "exposures.tobacco_smoking_status",
    "exposures.pack_years_smoked",
    "exposures.years_smoked",
]


def fetch_all_cases():
    """Fetch all TCGA-HNSC cases with clinical fields from GDC API."""

    filters = {
        "op": "=",
        "content": {
            "field": "project.project_id",
            "value": PROJECT_ID,
        },
    }

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(FIELDS),
        "format": "JSON",
        "size": "600",  # TCGA-HNSC has ~530 cases
    }

    print(f"Fetching clinical data for {PROJECT_ID}...")
    response = requests.get(GDC_CASES_ENDPOINT, params=params)
    response.raise_for_status()

    data = response.json()
    hits = data["data"]["hits"]
    total = data["data"]["pagination"]["total"]
    print(f"Retrieved {len(hits)} of {total} cases.")

    return hits


def flatten_case(case):
    """Flatten a single GDC case record into a flat dictionary."""

    row = {"case_id": case.get("submitter_id", "")}

    # Demographics
    demo = case.get("demographic", {})
    if isinstance(demo, list):
        demo = demo[0] if demo else {}
    row["gender"] = demo.get("gender")
    row["race"] = demo.get("race")
    row["ethnicity"] = demo.get("ethnicity")
    row["vital_status"] = demo.get("vital_status")
    row["days_to_death_demo"] = demo.get("days_to_death")
    row["year_of_birth"] = demo.get("year_of_birth")
    row["age_at_index"] = demo.get("age_at_index")

    # Diagnoses (take first if multiple)
    diagnoses = case.get("diagnoses", [])
    dx = diagnoses[0] if diagnoses else {}
    row["age_at_diagnosis_days"] = dx.get("age_at_diagnosis")
    row["primary_diagnosis"] = dx.get("primary_diagnosis")
    row["ajcc_pathologic_stage"] = dx.get("ajcc_pathologic_stage")
    row["ajcc_pathologic_t"] = dx.get("ajcc_pathologic_t")
    row["ajcc_pathologic_n"] = dx.get("ajcc_pathologic_n")
    row["ajcc_pathologic_m"] = dx.get("ajcc_pathologic_m")
    row["site_of_resection"] = dx.get("site_of_resection_or_biopsy")
    row["tissue_or_organ"] = dx.get("tissue_or_organ_of_origin")
    row["morphology"] = dx.get("morphology")
    row["vital_status_dx"] = dx.get("vital_status")
    row["days_to_last_follow_up"] = dx.get("days_to_last_follow_up")
    row["days_to_death_dx"] = dx.get("days_to_death")

    # Treatments (flatten into separate columns)
    treatments = dx.get("treatments", [])
    treatment_types = []
    for tx in treatments:
        tx_type = tx.get("treatment_type")
        if tx_type and tx_type not in treatment_types:
            treatment_types.append(tx_type)
    row["treatment_types"] = "; ".join(treatment_types) if treatment_types else None

    # Individual treatment flags
    all_tx = [t.get("treatment_type", "") for t in treatments]
    row["had_radiation"] = any("Radiation" in t for t in all_tx)
    row["had_pharmaceutical"] = any("Pharmaceutical" in t for t in all_tx)
    row["had_surgery"] = any("surg" in t.lower() for t in all_tx) if all_tx else False

    # Exposures
    exposures = case.get("exposures", [])
    exp = exposures[0] if exposures else {}
    row["alcohol_history"] = exp.get("alcohol_history")
    row["tobacco_smoking_status"] = exp.get("tobacco_smoking_status")
    row["pack_years_smoked"] = exp.get("pack_years_smoked")
    row["years_smoked"] = exp.get("years_smoked")

    return row


def main():
    hits = fetch_all_cases()
    rows = [flatten_case(case) for case in hits]
    df = pd.DataFrame(rows)

    # Save raw
    raw_path = OUTPUT_DIR / "tcga_hnsc_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample vital_status distribution:\n{df['vital_status'].value_counts()}")
    print(f"\nSample stage distribution:\n{df['ajcc_pathologic_stage'].value_counts()}")


if __name__ == "__main__":
    main()
