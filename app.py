"""
HNSC Clinical Decision Support Explorer

An interactive dashboard for exploring treatment strategies and patient
outcomes in Head and Neck Squamous Cell Carcinoma (HNSC) using TCGA data.

Features:
- Filter patients by clinical characteristics (including HPV proxy status)
- Compare outcomes across treatment strategies
- Find similar patient cohorts using Gower distance
- Kaplan-Meier survival analysis with log-rank tests
- Cox proportional hazards regression for adjusted outcome analysis

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from pathlib import Path

# ---- Page config ----
st.set_page_config(
    page_title="HNSC Clinical Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/tcga_hnsc_clean.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def plot_kaplan_meier(df, group_col, title="Overall Survival", time_col="survival_months"):
    """Generate a Kaplan-Meier plot comparing groups."""
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    groups = df[group_col].dropna().unique()
    groups = sorted(groups)

    for i, group in enumerate(groups):
        mask = df[group_col] == group
        subset = df[mask].dropna(subset=[time_col, "event_observed"])
        if len(subset) < 2:
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(
            subset[time_col],
            event_observed=subset["event_observed"],
            label=str(group),
        )

        timeline = kmf.survival_function_.index
        survival = kmf.survival_function_.iloc[:, 0]
        ci = kmf.confidence_interval_survival_function_

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=timeline,
                y=survival,
                mode="lines",
                name=f"{group} (n={len(subset)})",
                line=dict(color=color, width=2),
            )
        )

        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=list(timeline) + list(timeline[::-1]),
                y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1][::-1]),
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (months)",
        yaxis_title="Survival Probability",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=450,
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )

    return fig


def compute_gower_similarity(df, query):
    """
    Compute Gower similarity between a query patient profile and all patients.

    Gower distance handles mixed data types:
    - Categorical variables: 1 if match, 0 if mismatch
    - Continuous variables: 1 - |x_i - x_j| / range(x)

    Each variable is weighted by clinical relevance.
    Returns dataframe with similarity_score column (0 to 1).
    """
    categorical_features = {
        "stage_group": 3.0,
        "anatomic_site_group": 2.5,
        "gender": 1.0,
        "smoking_status": 1.5,
        "hpv_proxy": 2.0,
    }

    continuous_features = {
        "age_at_diagnosis_years": 2.0,
    }

    scores = np.zeros(len(df))
    total_weight = 0

    # Categorical: exact match = 1, mismatch = 0
    for feature, weight in categorical_features.items():
        if feature in query and query[feature] and feature in df.columns:
            match = (df[feature] == query[feature]).astype(float)
            scores += match * weight
            total_weight += weight

    # Continuous: 1 - normalized absolute distance
    for feature, weight in continuous_features.items():
        if feature in query and query[feature] is not None and feature in df.columns:
            col = df[feature].fillna(df[feature].median())
            feat_range = col.max() - col.min()
            if feat_range > 0:
                distance = np.abs(col - query[feature]) / feat_range
                similarity = 1.0 - distance
                scores += similarity * weight
                total_weight += weight

    if total_weight > 0:
        scores = scores / total_weight

    df = df.copy()
    df["similarity_score"] = scores
    return df


@st.cache_data
def run_cox_regression(df):
    """
    Fit a Cox Proportional Hazards model to estimate adjusted hazard ratios.

    Covariates: stage, anatomic site, age, gender, treatment, HPV proxy.
    Returns the fitted model summary as a dataframe and the model itself.
    """
    cox_df = df[["survival_months", "event_observed", "stage_group",
                  "anatomic_site_group", "age_at_diagnosis_years",
                  "gender", "treatment_strategy", "hpv_proxy"]].copy()

    cox_df = cox_df.dropna(subset=["survival_months", "event_observed", "age_at_diagnosis_years"])
    cox_df = cox_df[cox_df["survival_months"] > 0]

    # Filter out small categories (Cox needs enough events per group)
    for col in ["stage_group", "anatomic_site_group", "treatment_strategy"]:
        counts = cox_df[col].value_counts()
        valid = counts[counts >= 10].index
        cox_df = cox_df[cox_df[col].isin(valid)]

    # One-hot encode categorical variables
    cox_encoded = pd.get_dummies(
        cox_df,
        columns=["stage_group", "anatomic_site_group", "gender",
                  "treatment_strategy", "hpv_proxy"],
        drop_first=True,
        dtype=float,
    )

    # Clean column names (lifelines doesn't like special chars)
    cox_encoded.columns = [c.replace(" ", "_").replace("+", "plus").replace("/", "_") for c in cox_encoded.columns]

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        cox_encoded,
        duration_col="survival_months",
        event_col="event_observed",
    )

    summary = cph.summary[["coef", "exp(coef)", "se(coef)", "p", "exp(coef) lower 95%", "exp(coef) upper 95%"]].copy()
    summary = summary.rename(columns={
        "exp(coef)": "Hazard Ratio",
        "coef": "Coefficient",
        "se(coef)": "Std Error",
        "p": "p-value",
        "exp(coef) lower 95%": "HR Lower 95%",
        "exp(coef) upper 95%": "HR Upper 95%",
    })

    return summary, cph, len(cox_encoded)


def main():
    # ---- Load data ----
    if not DATA_PATH.exists():
        st.error(
            "Data file not found. Please run `python download_data.py` "
            "followed by `python process_data.py` first."
        )
        st.stop()

    df = load_data()

    # ---- Header ----
    st.title("🔬 HNSC Clinical Decision Support Explorer")
    st.markdown(
        "Interactive exploration of treatment strategies and patient outcomes "
        "in **Head and Neck Squamous Cell Carcinoma** using "
        "[TCGA-HNSC](https://portal.gdc.cancer.gov/projects/TCGA-HNSC) data."
    )

    # ---- Sidebar filters ----
    st.sidebar.header("🔍 Patient Filters")

    stage_options = sorted(df["stage_group"].dropna().unique())
    selected_stages = st.sidebar.multiselect(
        "Pathologic Stage", stage_options, default=stage_options
    )

    site_options = sorted(df["anatomic_site_group"].dropna().unique())
    selected_sites = st.sidebar.multiselect(
        "Anatomic Site", site_options, default=site_options
    )

    age_options = ["<45", "45-54", "55-64", "65-74", "75+"]
    age_available = [a for a in age_options if a in df["age_group"].values]
    selected_ages = st.sidebar.multiselect(
        "Age Group", age_available, default=age_available
    )

    gender_options = sorted(df["gender"].dropna().unique())
    selected_genders = st.sidebar.multiselect(
        "Gender", gender_options, default=gender_options
    )

    treatment_options = sorted(df["treatment_strategy"].dropna().unique())
    selected_treatments = st.sidebar.multiselect(
        "Treatment Strategy", treatment_options, default=treatment_options
    )

    hpv_options = sorted(df["hpv_proxy"].dropna().unique())
    selected_hpv = st.sidebar.multiselect(
        "HPV Status (site-based proxy)", hpv_options, default=hpv_options
    )

    # Apply filters
    mask = (
        df["stage_group"].isin(selected_stages)
        & df["anatomic_site_group"].isin(selected_sites)
        & df["age_group"].isin(selected_ages)
        & df["gender"].isin(selected_genders)
        & df["treatment_strategy"].isin(selected_treatments)
        & df["hpv_proxy"].isin(selected_hpv)
    )
    filtered = df[mask].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Patients in cohort", len(filtered))
    st.sidebar.metric("Events (deaths)", int(filtered["event_observed"].sum()))
    if len(filtered) > 0:
        st.sidebar.metric(
            "Median survival",
            f"{filtered['survival_months'].median():.1f} mo",
        )

    # ---- Tab layout ----
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Cohort Overview", "🔎 Find Similar Patients", "📈 Survival Analysis", "📉 Cox Regression"]
    )

    # ================================================================
    # TAB 1: Cohort Overview
    # ================================================================
    with tab1:
        if len(filtered) == 0:
            st.warning("No patients match the selected filters.")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", len(filtered))
        col2.metric("Median Age", f"{filtered['age_at_diagnosis_years'].median():.0f} yr")
        col3.metric(
            "Mortality Rate",
            f"{filtered['event_observed'].mean() * 100:.1f}%",
        )
        col4.metric(
            "Median Survival",
            f"{filtered['survival_months'].median():.1f} mo",
        )

        st.markdown("---")

        # Distribution charts
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            stage_counts = (
                filtered["stage_group"]
                .value_counts()
                .reindex(["Stage I", "Stage II", "Stage III", "Stage IV", "Not Reported"])
                .dropna()
            )
            fig_stage = px.bar(
                x=stage_counts.index,
                y=stage_counts.values,
                labels={"x": "Stage", "y": "Count"},
                title="Distribution by Stage",
                color=stage_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_stage.update_layout(showlegend=False, template="plotly_white", height=350)
            st.plotly_chart(fig_stage, width="stretch")

        with row1_col2:
            site_counts = filtered["anatomic_site_group"].value_counts()
            fig_site = px.bar(
                x=site_counts.index,
                y=site_counts.values,
                labels={"x": "Anatomic Site", "y": "Count"},
                title="Distribution by Anatomic Site",
                color=site_counts.index,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_site.update_layout(showlegend=False, template="plotly_white", height=350)
            st.plotly_chart(fig_site, width="stretch")

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            tx_counts = filtered["treatment_strategy"].value_counts()
            fig_tx = px.pie(
                names=tx_counts.index,
                values=tx_counts.values,
                title="Treatment Strategies",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_tx.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_tx, width="stretch")

        with row2_col2:
            fig_age = px.histogram(
                filtered,
                x="age_at_diagnosis_years",
                nbins=20,
                title="Age at Diagnosis",
                labels={"age_at_diagnosis_years": "Age (years)"},
                color_discrete_sequence=["#66c2a5"],
            )
            fig_age.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_age, width="stretch")

        # HPV proxy distribution
        row3_col1, row3_col2 = st.columns(2)

        with row3_col1:
            hpv_counts = filtered["hpv_proxy"].value_counts()
            fig_hpv = px.pie(
                names=hpv_counts.index,
                values=hpv_counts.values,
                title="HPV Status (Site-Based Proxy)",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_hpv.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_hpv, width="stretch")

        with row3_col2:
            smoking_counts = filtered["smoking_status"].value_counts()
            fig_smoking = px.bar(
                x=smoking_counts.index,
                y=smoking_counts.values,
                labels={"x": "Smoking Status", "y": "Count"},
                title="Tobacco Smoking Status",
                color=smoking_counts.index,
                color_discrete_sequence=px.colors.qualitative.Pastel2,
            )
            fig_smoking.update_layout(showlegend=False, template="plotly_white", height=380)
            st.plotly_chart(fig_smoking, width="stretch")

    # ================================================================
    # TAB 2: Find Similar Patients (Gower Distance)
    # ================================================================
    with tab2:
        st.subheader("Find Comparable Patients")
        st.markdown(
            "Define a patient profile to find the most similar cohort using "
            "**Gower distance** (weighted similarity across categorical and continuous features). "
            "Then explore their treatment outcomes and survival."
        )

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            q_stage = st.selectbox(
                "Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"], index=2
            )
            q_site = st.selectbox(
                "Anatomic Site",
                sorted([s for s in df["anatomic_site_group"].unique() if s != "Other/Unknown"]),
            )
            q_age = st.slider(
                "Age at Diagnosis",
                min_value=int(df["age_at_diagnosis_years"].min()),
                max_value=int(df["age_at_diagnosis_years"].max()),
                value=60,
                help="Continuous matching: patients closer in age score higher similarity.",
            )

        with sim_col2:
            q_gender = st.selectbox("Gender", sorted(df["gender"].dropna().unique()))
            q_smoking = st.selectbox(
                "Smoking Status", sorted(df["smoking_status"].dropna().unique())
            )
            q_hpv = st.selectbox(
                "HPV Status (proxy)",
                ["Likely HPV-associated", "Likely HPV-negative"],
            )
            min_similarity = st.slider(
                "Minimum similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher values return fewer but more similar patients.",
            )

        query = {
            "stage_group": q_stage,
            "anatomic_site_group": q_site,
            "age_at_diagnosis_years": q_age,
            "gender": q_gender,
            "smoking_status": q_smoking,
            "hpv_proxy": q_hpv,
        }

        df_scored = compute_gower_similarity(df, query)
        similar = df_scored[df_scored["similarity_score"] >= min_similarity].copy()

        st.markdown("---")

        if len(similar) < 5:
            st.warning(
                f"Only {len(similar)} patients match at this threshold. "
                "Consider lowering the minimum similarity."
            )

        if len(similar) > 0:
            sim_m1, sim_m2, sim_m3, sim_m4 = st.columns(4)
            sim_m1.metric("Matching Patients", len(similar))
            sim_m2.metric(
                "Median Survival",
                f"{similar['survival_months'].median():.1f} mo",
            )
            sim_m3.metric(
                "Mortality Rate",
                f"{similar['event_observed'].mean() * 100:.1f}%",
            )
            sim_m4.metric(
                "Avg Similarity",
                f"{similar['similarity_score'].mean():.2f}",
            )

            # Treatment outcomes for similar patients
            st.markdown("#### Treatment Outcomes in Similar Cohort")

            tx_summary = (
                similar.groupby("treatment_strategy")
                .agg(
                    n_patients=("case_id", "count"),
                    median_survival_mo=("survival_months", "median"),
                    mortality_pct=("event_observed", lambda x: f"{x.mean() * 100:.1f}%"),
                )
                .sort_values("n_patients", ascending=False)
                .reset_index()
            )
            tx_summary.columns = [
                "Treatment Strategy",
                "Patients",
                "Median Survival (mo)",
                "Mortality (%)",
            ]
            st.dataframe(tx_summary, width="stretch", hide_index=True)

            # KM for similar cohort by treatment
            if similar["treatment_strategy"].nunique() > 1:
                fig_sim_km = plot_kaplan_meier(
                    similar,
                    "treatment_strategy",
                    title="Survival by Treatment (Similar Patients)",
                )
                st.plotly_chart(fig_sim_km, width="stretch")

            # Comparison: similar vs rest
            st.markdown("#### Similar Cohort vs. All Other Patients")
            df_compare = df.copy()
            df_compare["cohort"] = np.where(
                df_compare.index.isin(similar.index),
                "Similar Cohort",
                "Other Patients",
            )
            fig_compare = plot_kaplan_meier(
                df_compare, "cohort", title="Similar Cohort vs. All Others"
            )
            st.plotly_chart(fig_compare, width="stretch")

            # Log-rank test
            sim_data = similar.dropna(subset=["survival_months", "event_observed"])
            other_data = df_compare[df_compare["cohort"] == "Other Patients"].dropna(
                subset=["survival_months", "event_observed"]
            )
            if len(sim_data) >= 5 and len(other_data) >= 5:
                result = logrank_test(
                    sim_data["survival_months"],
                    other_data["survival_months"],
                    event_observed_A=sim_data["event_observed"],
                    event_observed_B=other_data["event_observed"],
                )
                st.caption(
                    f"Log-rank test p-value: {result.p_value:.4f} "
                    f"({'statistically significant' if result.p_value < 0.05 else 'not statistically significant'} "
                    f"at α=0.05)"
                )

    # ================================================================
    # TAB 3: Survival Analysis
    # ================================================================
    with tab3:
        st.subheader("Survival Analysis")
        st.markdown("Compare Kaplan-Meier survival curves across clinical subgroups.")

        compare_by = st.selectbox(
            "Compare survival by",
            [
                "stage_group",
                "anatomic_site_group",
                "treatment_strategy",
                "hpv_proxy",
                "age_group",
                "gender",
                "smoking_status",
                "n_stage",
            ],
            format_func=lambda x: {
                "stage_group": "Pathologic Stage",
                "anatomic_site_group": "Anatomic Site",
                "treatment_strategy": "Treatment Strategy",
                "hpv_proxy": "HPV Status (Proxy)",
                "age_group": "Age Group",
                "gender": "Gender",
                "smoking_status": "Smoking Status",
                "n_stage": "Nodal Status (N)",
            }.get(x, x),
        )

        nice_name = {
            "stage_group": "Pathologic Stage",
            "anatomic_site_group": "Anatomic Site",
            "treatment_strategy": "Treatment Strategy",
            "hpv_proxy": "HPV Status (Proxy)",
            "age_group": "Age Group",
            "gender": "Gender",
            "smoking_status": "Smoking Status",
            "n_stage": "Nodal Status (N)",
        }.get(compare_by, compare_by)

        fig_km = plot_kaplan_meier(
            filtered, compare_by, title=f"Overall Survival by {nice_name}"
        )
        st.plotly_chart(fig_km, width="stretch")

        # Summary table
        st.markdown(f"#### Summary by {nice_name}")
        summary = (
            filtered.groupby(compare_by)
            .agg(
                n=("case_id", "count"),
                median_survival=("survival_months", "median"),
                mean_age=("age_at_diagnosis_years", "mean"),
                mortality=("event_observed", "mean"),
            )
            .round(1)
            .reset_index()
        )
        summary["mortality"] = (summary["mortality"] * 100).round(1).astype(str) + "%"
        summary.columns = [nice_name, "N", "Median Survival (mo)", "Mean Age", "Mortality"]
        st.dataframe(summary, width="stretch", hide_index=True)

    # ================================================================
    # TAB 4: Cox Proportional Hazards Regression
    # ================================================================
    with tab4:
        st.subheader("Cox Proportional Hazards Regression")
        st.markdown(
            "Estimates the **adjusted** effect of each clinical variable on survival, "
            "controlling for all other variables simultaneously. "
            "A **hazard ratio > 1** means higher risk of death; **< 1** means protective."
        )

        try:
            cox_summary, cph, n_cox = run_cox_regression(filtered)

            st.caption(f"Model fitted on {n_cox} patients with complete data.")

            # Clean up variable names for display
            display_df = cox_summary.copy()
            display_df.index = (
                display_df.index
                .str.replace("stage_group_", "Stage: ", regex=False)
                .str.replace("anatomic_site_group_", "Site: ", regex=False)
                .str.replace("treatment_strategy_", "Treatment: ", regex=False)
                .str.replace("hpv_proxy_", "HPV: ", regex=False)
                .str.replace("gender_", "Gender: ", regex=False)
                .str.replace("age_at_diagnosis_years", "Age (per year)", regex=False)
                .str.replace("_", " ", regex=False)
                .str.replace("plus", "+", regex=False)
            )
            display_df = display_df.round(3)

            st.dataframe(display_df, width="stretch")

            # Forest plot
            st.markdown("#### Forest Plot (Hazard Ratios)")
            plot_df = display_df.reset_index()
            plot_df.columns = ["Variable", "Coefficient", "HR", "SE", "p-value", "HR_lower", "HR_upper"]
            plot_df = plot_df.sort_values("HR", ascending=True)

            fig_forest = go.Figure()

            # Confidence intervals
            fig_forest.add_trace(
                go.Scatter(
                    x=plot_df["HR"],
                    y=plot_df["Variable"],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=["#d62728" if p < 0.05 else "#7f7f7f" for p in plot_df["p-value"]],
                    ),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(plot_df["HR_upper"] - plot_df["HR"]).tolist(),
                        arrayminus=(plot_df["HR"] - plot_df["HR_lower"]).tolist(),
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "HR: %{x:.2f}<br>"
                        "95% CI: %{customdata[0]:.2f} – %{customdata[1]:.2f}<br>"
                        "p: %{customdata[2]:.3f}<extra></extra>"
                    ),
                    customdata=plot_df[["HR_lower", "HR_upper", "p-value"]].values,
                )
            )

            # Reference line at HR = 1
            fig_forest.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)

            fig_forest.update_layout(
                title="Adjusted Hazard Ratios with 95% Confidence Intervals",
                xaxis_title="Hazard Ratio",
                template="plotly_white",
                height=max(400, len(plot_df) * 35),
                showlegend=False,
                margin=dict(l=250),
            )

            st.plotly_chart(fig_forest, width="stretch")

            st.markdown(
                "**How to read this:** Each point is the hazard ratio for that variable, "
                "adjusted for all other variables. The horizontal line is the 95% confidence interval. "
                "**Red** points are statistically significant (p < 0.05). "
                "If the confidence interval crosses the dashed line at 1.0, "
                "the effect is not statistically significant. "
                "For example, a hazard ratio of 2.0 for Stage IV means Stage IV patients "
                "have roughly twice the risk of death compared to the reference category, "
                "after controlling for age, site, treatment, and other factors."
            )

            st.markdown(
                "**Note:** A lower hazard ratio for a treatment strategy does not necessarily "
                "mean that treatment is better. Treatment selection is not randomized in this data; "
                "patients who received certain treatments may have differed systematically from "
                "those who received others (selection bias). These results are exploratory, "
                "not causal."
            )

        except Exception as e:
            st.error(
                f"Cox regression could not be fitted on the current filter selection. "
                f"Try broadening your filters to include more patients. Error: {e}"
            )

    # ---- Footer ----
    st.markdown("---")
    st.caption(
        "Data source: [TCGA-HNSC](https://portal.gdc.cancer.gov/projects/TCGA-HNSC) "
        "via the GDC API. HPV status is approximated using anatomic site as a proxy "
        "(oropharyngeal tumors classified as likely HPV-associated). "
        "This tool is for research and educational purposes only "
        "and is not intended for clinical decision-making. "
        "Built by [Avani Bhat](https://github.com/YOUR_GITHUB_USERNAME)."
    )


if __name__ == "__main__":
    main()
