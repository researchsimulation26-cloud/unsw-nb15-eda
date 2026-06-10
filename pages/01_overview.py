import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dataset Overview - UNSW-NB15", page_icon="📊", layout="wide")

from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    horizontal_bar,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import COLORS, NUMERIC_COLS, CATEGORICAL_COLS, FOOTER

style_metric_cards()

df = setup_sidebar()

if df.empty:
    st.error("Dataset not loaded. Check data files.")
    st.stop()

st.title("📊 Dataset Overview")
st.markdown(
    "This page provides a high-level summary of the dataset structure, feature types, "
    "missing values, and a preview of the raw records."
)

total_rows = len(df)
total_cols = df.shape[1]
memory_mb = df.memory_usage(deep=True).sum() / 1024**2
numeric_count = df.select_dtypes(include=[np.number]).shape[1]
categorical_count = df.select_dtypes(include=["str", "category"]).shape[1]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Rows", f"{total_rows:,}")
k2.metric("Total Columns", total_cols)
k3.metric("Memory Usage", f"{memory_mb:.1f} MB")
k4.metric("Numeric Features", numeric_count)
k5.metric("Categorical Features", categorical_count)
st.markdown("---")

st.subheader("📋 Basic Statistics")
with st.expander("View Descriptive Statistics", expanded=False):
    desc_df = df.describe(include="all").T
    desc_df = desc_df.reset_index().rename(columns={"index": "Feature"})
    st.dataframe(desc_df, width="stretch", height=400)
    download_csv_button(desc_df, "descriptive_statistics.csv", "Download Statistics")

st.markdown("---")

st.subheader("🏷️ Data Types Distribution")
dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
dtype_counts.columns = ["Data Type", "Count"]
dtype_counts = dtype_counts.sort_values("Count", ascending=True)
fig = horizontal_bar(
    dtype_counts, y="Data Type", x="Count",
    title="Feature Data Type Distribution",
    color="#00d4ff", sort_values=True, height=300,
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Numeric features dominate the dataset, reflecting the quantitative nature of "
    "network traffic measurements (byte counts, packet counts, timing, etc.). "
    "Categorical features include protocol types, service names, and connection states."
)

st.markdown("---")

st.subheader("🧩 Column Information")
col_info = pd.DataFrame(
    {
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Unique Count": [df[c].nunique() for c in df.columns],
        "Null Count": [df[c].isna().sum() for c in df.columns],
        "Null %": [f"{df[c].isna().mean() * 100:.2f}%" for c in df.columns],
    }
)
col_info.index = range(1, len(col_info) + 1)
st.dataframe(col_info, width="stretch", height=500)
download_csv_button(col_info, "column_info.csv", "Download Column Info")

st.markdown("---")

st.subheader("⚠️ Missing Values Analysis")
null_counts = df.isna().sum()
null_cols = null_counts[null_counts > 0]
if len(null_cols) > 0:
    null_df = null_cols.reset_index()
    null_df.columns = ["Feature", "Missing Count"]
    null_df["Missing %"] = (null_df["Missing Count"] / total_rows * 100).round(2)
    null_df["Missing %"] = null_df["Missing %"].apply(lambda x: f"{x:.2f}%")

    fig = horizontal_bar(
        null_df, y="Feature", x="Missing Count",
        title="Columns with Missing Values",
        color="#ff6b35", sort_values=True, height=300,
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Why are these columns missing data?**\n\n"
        "- **`attack_cat`**: Missing values represent **Normal** traffic. The dataset assigns "
        "an attack category only when `Label=1`. Normal records have no attack category.\n"
        "- **`ct_flw_http_mthd`**: Only populated for HTTP flows. Non-HTTP traffic (DNS, SSH, FTP, "
        "etc.) does not have HTTP method information, resulting in NaN.\n"
        "- **`is_ftp_login`**: Only relevant for FTP connections where a login attempt occurs. "
        "Non-FTP traffic has no login information.\n\n"
        "These missing values are handled by filling with appropriate defaults (Normal, 0, 0)."
    )
else:
    st.success("✅ No missing values found in the dataset.")

st.markdown("---")

st.markdown("### ✅ Column Health Check")
health_data = []
for c in df.columns:
    null_pct = df[c].isna().mean() * 100
    health_data.append(
        {
            "Column": c,
            "Nulls %": f"{null_pct:.2f}%",
            "Status": "✅" if null_pct == 0 else "⚠️",
        }
    )
health_df = pd.DataFrame(health_data)
st.dataframe(health_df, width="stretch", height=300)
download_csv_button(health_df, "column_health.csv", "Download Health Check")

st.markdown("---")

st.subheader("👁️ Sample Data Viewer")
view_mode = st.radio("View Mode", ["All Records", "Attack Only"], horizontal=True)
if view_mode == "Attack Only":
    sample_df = df[df["Label"] == 1].head(100)
else:
    sample_df = df.head(100)

attack_cat_filter = st.multiselect(
    "Filter by Attack Category",
    options=sorted(df["attack_cat"].dropna().unique()),
    default=[],
)
if attack_cat_filter:
    sample_df = sample_df[sample_df["attack_cat"].isin(attack_cat_filter)]

st.dataframe(sample_df, width="stretch", height=400)
download_csv_button(sample_df, "sample_data.csv", "Download Sample")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)
