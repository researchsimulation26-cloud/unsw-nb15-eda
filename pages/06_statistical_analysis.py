import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Statistical Analysis - UNSW-NB15", page_icon="📈", layout="wide")

from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    horizontal_bar,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import COLORS, NUMERIC_COLS, FOOTER

style_metric_cards()

df = setup_sidebar()

if df.empty:
    st.error("Dataset not loaded.")
    st.stop()

st.title("📈 Statistical Analysis")
st.markdown(
    "Statistical hypothesis testing, feature importance estimation, and outlier analysis "
    "to quantify the discriminative power of each feature for attack detection."
)

num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Label"]]
num_cols = [c for c in num_cols if df[c].nunique() > 2]
num_cols = num_cols[:30]

n_features = len(num_cols)
n_normal = len(df[df["Label"] == 0])
n_attack = len(df[df["Label"] == 1])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Numeric Features Tested", n_features)
k2.metric("Normal Samples", f"{n_normal:,}")
k3.metric("Attack Samples", f"{n_attack:,}")
k4.metric("Total Samples", f"{len(df):,}")
st.markdown("---")

st.subheader("🧪 Mann-Whitney U Test: Normal vs Attack")
st.markdown(
    "The Mann-Whitney U test (non-parametric) evaluates whether the distributions of "
    "each feature differ significantly between Normal and Attack groups."
)

if st.button("Run Mann-Whitney U Tests", type="primary", use_container_width=True):
    with st.spinner("Running statistical tests..."):
        results = []
        normal = df[df["Label"] == 0]
        attack = df[df["Label"] == 1]

        for col in num_cols:
            n_vals = normal[col].dropna().values
            a_vals = attack[col].dropna().values
            if len(n_vals) < 5 or len(a_vals) < 5:
                continue
            try:
                stat, p = mannwhitneyu(n_vals, a_vals, alternative="two-sided")
                n_mean = np.mean(n_vals)
                a_mean = np.mean(a_vals)
                n_total = len(n_vals) + len(a_vals)
                r = 1 - (2 * stat) / (len(n_vals) * len(a_vals))
                results.append(
                    {
                        "Feature": col,
                        "Normal Mean": f"{n_mean:.4f}",
                        "Attack Mean": f"{a_mean:.4f}",
                        "U-statistic": f"{stat:.1f}",
                        "p-value": f"{p:.6e}",
                        "Significant": "✅" if p < 0.05 else "❌",
                        "Effect Size (r)": f"{r:.4f}",
                    }
                )
            except Exception:
                continue

        mw_df = pd.DataFrame(results)
        mw_df["abs_r"] = mw_df["Effect Size (r)"].str.replace(",", ".").astype(float).abs()
        mw_df = mw_df.sort_values("abs_r", ascending=False).drop(columns=["abs_r"])

        st.dataframe(mw_df, use_container_width=True, height=500)
        download_csv_button(mw_df, "mann_whitney_results.csv", "Download Test Results")

        sig_count = (mw_df["Significant"] == "✅").sum()
        st.success(
            f"✅ {sig_count}/{len(mw_df)} features show statistically significant "
            f"differences (p < 0.05) between Normal and Attack traffic."
        )
        st.caption(
            "The effect size (rank-biserial correlation, r) quantifies the magnitude of difference. "
            "Features with |r| > 0.5 have large effects and are strong univariate discriminators "
            "between normal and attack traffic."
        )

st.markdown("---")

st.subheader("🌲 Feature Importance (Random Forest)")
st.markdown(
    "A Random Forest classifier is trained on a 50k sample to estimate feature importance "
    "for predicting whether traffic is normal or attack."
)

if st.button("Compute Random Forest Importance", type="primary", use_container_width=True):
    with st.spinner("Training Random Forest on 50k sample..."):
        sample_rf = df.sample(n=min(50000, len(df)))
        rf_features = [c for c in num_cols if c in sample_rf.columns]
        X_rf = sample_rf[rf_features].fillna(0)
        y_rf = sample_rf["Label"]

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            random_state=42, n_jobs=-1,
            verbose=0,
        )
        rf.fit(X_rf, y_rf)

        rf_importance = pd.DataFrame(
            {"Feature": rf_features, "Importance": rf.feature_importances_}
        ).sort_values("Importance", ascending=True).tail(20)

        fig = horizontal_bar(
            rf_importance, y="Feature", x="Importance",
            title="Top 20 Feature Importances (Random Forest)",
            color="#00d4ff", sort_values=False, height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Random Forest importance reflects how much each feature contributes to "
            "reducing impurity in decision tree splits. Top features are most informative "
            "for predicting attack traffic. Note: this is for exploratory purposes only, "
            "not a production-ready model."
        )
        download_csv_button(
            rf_importance.sort_values("Importance", ascending=False),
            "rf_feature_importance.csv",
            "Download RF Importance",
        )

st.markdown("---")

st.subheader("📊 Mutual Information Scores")
st.markdown(
    "Mutual Information (MI) measures the dependence between each feature and the Label, "
    "capturing non-linear relationships that correlation may miss."
)

if st.button("Compute Mutual Information", type="primary", use_container_width=True):
    with st.spinner("Computing mutual information..."):
        sample_mi = df.sample(n=min(100000, len(df)))
        mi_features = [c for c in num_cols if c in sample_mi.columns]
        X_mi = sample_mi[mi_features].fillna(0)
        y_mi = sample_mi["Label"]

        mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
        mi_df = pd.DataFrame(
            {"Feature": mi_features, "Mutual Information": mi_scores}
        ).sort_values("Mutual Information", ascending=True).tail(20)

        fig = horizontal_bar(
            mi_df, y="Feature", x="Mutual Information",
            title="Top 20 Features by Mutual Information with Label",
            color="#ff6b35", sort_values=False, height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Mutual Information captures both linear and non-linear dependencies. "
            "Comparing MI with Random Forest importance reveals which features are "
            "consistently informative across different metrics. Disagreements may "
            "indicate features with non-linear predictive power."
        )
        download_csv_button(
            mi_df.sort_values("Mutual Information", ascending=False),
            "mutual_information.csv",
            "Download MI Scores",
        )

st.markdown("---")

st.subheader("⚠️ Outlier Analysis")
st.markdown(
    "Outliers in network data often represent actual attack traffic. This analysis "
    "identifies features with the most extreme values (beyond 3 standard deviations)."
)

if st.button("Run Outlier Analysis", type="primary", use_container_width=True):
    with st.spinner("Analyzing outliers..."):
        var_cols = df[num_cols].var().sort_values(ascending=False).head(10).index
        outlier_results = []

        for col in var_cols:
            vals = df[col].dropna()
            mean, std = vals.mean(), vals.std()
            if std == 0:
                continue
            outliers = vals[(vals - mean).abs() > 3 * std]
            outlier_results.append(
                {
                    "Feature": col,
                    "Total Outliers": len(outliers),
                    "% Outliers": f"{len(outliers) / len(vals) * 100:.2f}%",
                    "Min": f"{vals.min():.4f}",
                    "Max": f"{vals.max():.4f}",
                    "Mean": f"{mean:.4f}",
                    "Std": f"{std:.4f}",
                }
            )

        outlier_df = pd.DataFrame(outlier_results)
        st.dataframe(outlier_df, use_container_width=True)
        download_csv_button(outlier_df, "outlier_analysis.csv", "Download Outlier Analysis")

        st.caption(
            "Features with high variance and many outliers are often strong attack indicators. "
            "For example, extreme `sbytes` values may represent large exploit payloads, while "
            "near-zero `dur` with high `Spkts` suggests DoS flooding behaviour. In network "
            "intrusion detection, outliers are often the most interesting data points."
        )

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)
