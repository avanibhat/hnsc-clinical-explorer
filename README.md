---
title: HNSC Clinical Explorer
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

\

# HNSC Clinical Decision Support Explorer

An interactive dashboard for exploring treatment strategies and patient outcomes in **Head and Neck Squamous Cell Carcinoma (HNSC)** using publicly available data from [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/projects/TCGA-HNSC).

Built with Streamlit, Plotly, and lifelines.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open in HF Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/avanibhat/hnsc-clinical-explorer)

## Motivation

Clinicians treating head and neck cancers often need to compare treatment outcomes across patient cohorts with similar clinical characteristics. Published clinical trial results may not represent a specific patient's profile, and institutional data is rarely available in an interactive, explorable format.

This tool enables data-driven exploration of treatment strategies by allowing users to filter patients by stage, anatomic site, age, HPV status, and other clinical variables, then examine survival outcomes for comparable groups. It supports three analytical workflows and one inferential model:

1. **Cohort Overview**: Visualize the distribution of clinical characteristics across a filtered patient set, including HPV proxy status and smoking exposure.
2. **Find Similar Patients**: Define a patient profile and retrieve the most similar cohort using Gower distance (weighted similarity across categorical and continuous variables), along with their treatment outcomes and survival curves.
3. **Survival Analysis**: Compare Kaplan-Meier survival curves across clinical subgroups with log-rank significance testing.
4. **Cox Regression**: Estimate adjusted hazard ratios for clinical variables using Cox Proportional Hazards modelling, controlling for confounders. Includes a forest plot for visual interpretation.

## Clinical Context

Head and neck squamous cell carcinoma encompasses cancers of the oral cavity, oropharynx, hypopharynx, and larynx. A critical distinction in this disease is HPV status: HPV-associated oropharyngeal cancers (primarily tonsil and base of tongue) have substantially better prognosis than HPV-negative cancers at other sites. Because molecular HPV testing data is not available in the standard TCGA clinical download, this tool uses anatomic site as a proxy, classifying oropharyngeal tumors as "Likely HPV-associated." This proxy is well established in the literature (Gillison et al., JAMA 2000; Chaturvedi et al., JCO 2011) though less precise than molecular testing.

Treatment strategies in HNSC typically involve surgery, radiation, chemotherapy, or combinations thereof, and the optimal approach varies by stage, site, and HPV status. This tool allows exploration of how these strategies correlate with survival outcomes across different patient profiles.

## Data

All data is sourced from the [GDC Data Portal](https://portal.gdc.cancer.gov/) (open access, no authentication required). The dataset includes clinical, demographic, exposure, and treatment information for approximately 530 TCGA-HNSC patients.

Key variables include pathologic stage (AJCC), anatomic site, treatment modalities (surgery, radiation, chemotherapy), tobacco and alcohol exposure, and overall survival.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hnsc-clinical-explorer.git
cd hnsc-clinical-explorer
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download and process the data

```bash
python download_data.py
python process_data.py
```

This fetches clinical data from the GDC API and produces a cleaned CSV in `data/tcga_hnsc_clean.csv`.

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
hnsc-clinical-explorer/
├── app.py                 # Streamlit dashboard (KM, Cox, Gower similarity)
├── download_data.py       # Fetch TCGA-HNSC data from GDC API
├── process_data.py        # Clean data, derive features, HPV proxy
├── requirements.txt       # Python dependencies
├── data/
│   ├── tcga_hnsc_raw.csv      # Raw download (gitignored)
│   └── tcga_hnsc_clean.csv    # Processed dataset
├── .gitignore
└── README.md
```

## Features

- **Interactive filtering** by pathologic stage, anatomic site, age group, gender, treatment strategy, and HPV proxy status
- **Comparable cohort matching** using Gower distance with weighted similarity across categorical (stage, site, gender, smoking, HPV) and continuous (age) features
- **Kaplan-Meier survival curves** with confidence intervals and log-rank tests
- **Cox Proportional Hazards regression** with adjusted hazard ratios and forest plot visualization
- **HPV status proxy** derived from anatomic site, reflecting the strong association between oropharyngeal tumors and HPV
- **Treatment outcome comparison** tables for matched patient groups
- All visualizations are interactive (Plotly) and exportable

## Limitations

This tool is intended for **research and educational purposes only** and is not a clinical decision-making instrument. Key limitations:

- TCGA data overrepresents patients treated at academic medical centers and may not reflect broader populations.
- Treatment information may be incomplete; "No treatment recorded" may indicate missing data rather than absence of treatment.
- HPV status is approximated by anatomic site, not by molecular testing. This proxy has known false positive and false negative rates.
- Cox regression results are observational and subject to selection bias. Treatment effects are not causal estimates.
- Approximately 22% of patients have unreported pathologic stage.

## License

MIT

## Author

Avani Bhat — [LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) | [GitHub](https://github.com/YOUR_GITHUB_USERNAME)
