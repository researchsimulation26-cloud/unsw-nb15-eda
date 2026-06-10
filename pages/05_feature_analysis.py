import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Feature Analysis - UNSW-NB15", page_icon="🔬", layout="wide")

from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    histogram,
    box_plot,
    heatmap,
    horizontal_bar,
    stacked_bar,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import (
    ATTACK_COLORS, COLORS, FOOTER,
    NUMERIC_COLS, TCP_FEATURES, CT_FEATURES,
)

style_metric_cards()

df = setup_sidebar()

if df.empty:
    st.error("Dataset not loaded.")
    st.stop()

st.title("🔬 Feature Analysis")
st.markdown(
    "Explore individual feature distributions, correlations, and relationships with "
    "the target label. This page helps identify the most informative features for "
    "intrusion detection modelling."
)

numeric_df = df.select_dtypes(include=[np.number])
available_numeric = [c for c in numeric_df.columns if c in NUMERIC_COLS or c in df.columns]

total_features = df.shape[1]
numeric_count = numeric_df.shape[1]
has_tcp = all(f in df.columns for f in TCP_FEATURES)
has_ct = all(f in df.columns for f in CT_FEATURES[:6])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Features", total_features)
k2.metric("Numeric Features", numeric_count)
k3.metric("Has TCP Features", "✅" if has_tcp else "❌")
k4.metric("Has CT Features", "✅" if has_ct else "❌")
st.markdown("---")

st.subheader("📈 Feature Distribution Explorer")
num_features = [c for c in numeric_df.columns if c not in ["Label", "is_sm_ips_ports"]]
selected_feat = st.selectbox("Select a numeric feature", sorted(num_features))

sample_feat = df.sample(n=min(200000, len(df))) if len(df) > 200000 else df
sample_feat["label_name"] = sample_feat["Label"].map({0: "Normal", 1: "Attack"})

use_log = st.checkbox("Enable log scale", value=True, key="feat_log")
col1, col2 = st.columns([2, 1])

with col1:
    fig = histogram(
        sample_feat, x=selected_feat, color="label_name",
        title=f"{selected_feat} Distribution: Normal vs Attack",
        log_x=use_log,
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    stats = sample_feat.groupby("label_name")[selected_feat].describe()
    st.dataframe(stats, use_container_width=True)

if len(df) > 200000:
    st.info("📌 Chart based on 200k sample for performance")

st.caption(
    f"The histogram shows the distribution of `{selected_feat}` for both Normal and Attack traffic. "
    f"Differences in distribution shape, central tendency, or spread indicate the feature's "
    f"discriminative power for intrusion detection."
)

st.markdown("---")

st.subheader("🔗 Correlation Heatmap")
st.markdown("### Feature Selection")
corr_features = st.multiselect(
    "Select features for correlation matrix",
    options=sorted(num_features),
    default=sorted(num_features[:min(20, len(num_features))]),
)
corr_features = [c for c in corr_features if c in numeric_df.columns]

if len(corr_features) >= 2:
    corr_df = numeric_df[corr_features].dropna().corr()
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    corr_display = corr_df.mask(mask)

    fig = heatmap(
        corr_display,
        title="Feature Correlation Matrix (Lower Triangle)",
        colorscale="RdBu_r",
        height=max(500, 25 * len(corr_features)),
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Highly Correlated Pairs")
    corr_pairs = (
        corr_df.where(~np.eye(corr_df.shape[0], dtype=bool))
        .unstack()
        .reset_index()
    )
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
    corr_pairs["Abs Correlation"] = corr_pairs["Correlation"].abs()
    corr_pairs = corr_pairs.sort_values("Abs Correlation", ascending=False).head(10)
    corr_pairs = corr_pairs[corr_pairs["Feature 1"] != corr_pairs["Feature 2"]]
    st.dataframe(corr_pairs, use_container_width=True)
    download_csv_button(corr_pairs, "top_correlated_pairs.csv", "Download Correlated Pairs")

    st.caption(
        "High correlations between features indicate redundancy. In modelling, highly "
        "correlated features may not add independent predictive value. Pairs like "
        "`sbytes`-`Spkts` or `tcprtt`-`synack` are often correlated as they measure "
        "related aspects of network flows."
    )
else:
    st.warning("Select at least 2 features to compute correlations.")

st.markdown("---")

st.subheader("🎯 Feature Correlation with Label")
label_corr = (
    numeric_df.corrwith(df["Label"])
    .dropna()
    .sort_values(key=abs, ascending=False)
    .reset_index()
)
label_corr.columns = ["Feature", "Correlation with Label"]
label_corr["Abs Correlation"] = label_corr["Correlation with Label"].abs()

label_corr["Color"] = label_corr["Correlation with Label"].apply(
    lambda x: "#00d4ff" if x > 0 else "#f85149"
)

fig = horizontal_bar(
    label_corr.head(20),
    y="Feature",
    x="Correlation with Label",
    title="Top 20 Features by Absolute Correlation with Label",
    color="#00d4ff",
    sort_values=True,
    height=500,
)
fig.update_traces(
    marker_color=[
        "#00d4ff" if v > 0 else "#f85149"
        for v in label_corr.head(20)["Correlation with Label"]
    ]
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Features with high absolute correlation to the Label are strong univariate predictors. "
    "Blue bars indicate positive correlation (higher values → higher attack probability), "
    "while red bars indicate negative correlation. Note that correlation only measures "
    "linear relationships — non-linear predictors may show low correlation despite being useful."
)
download_csv_button(label_corr, "feature_label_correlation.csv", "Download Correlations")

st.markdown("---")

st.subheader("🔌 TCP Feature Analysis")
if has_tcp:
    tcp_sample = df.sample(n=min(100000, len(df))) if len(df) > 100000 else df
    tcp_features_present = [f for f in TCP_FEATURES if f in tcp_sample.columns]
    for i in range(0, len(tcp_features_present), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(tcp_features_present):
                feat = tcp_features_present[idx]
                fig = box_plot(
                    tcp_sample, x="attack_cat", y=feat,
                    title=f"{feat} by Attack Category",
                    log_y=True,
                    height=400,
                )
                with cols[j]:
                    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "TCP features like window size (`swin`, `dwin`) and RTT (`tcprtt`, `synack`, `ackdat`) "
        "reveal connection behaviour. Attack traffic often shows atypical TCP parameters — "
        "for example, unusually small window sizes may indicate scanning tools, while "
        "abnormal RTT patterns may reveal spoofed or manipulated connections."
    )
else:
    st.info("ℹ️ TCP feature analysis is available in all datasets.")

st.markdown("---")

st.subheader("🔗 Connection Tracking Features")
if has_ct:
    ct_present = [f for f in CT_FEATURES if f in df.columns]
    ct_means = df.groupby("attack_cat")[ct_present].mean().reset_index()
    ct_melted = ct_means.melt(id_vars=["attack_cat"], var_name="Feature", value_name="Mean")
    ct_melted = ct_melted[ct_melted["attack_cat"] != "Normal"]

    fig = stacked_bar(
        ct_melted, x="attack_cat", y="Mean", color="Feature",
        title="Mean Connection Tracking Features per Attack Category",
        height=500,
        color_map=None,
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Connection tracking features (`ct_srv_src`, `ct_dst_ltm`, etc.) capture "
        "relational patterns like \"how many connections to the same service from this source?\". "
        "High `ct_srv_src` values are indicative of scanner behaviour — a single source "
        "probing the same service across many destinations."
    )

    ct_table = ct_means.round(2)
    st.dataframe(ct_table, use_container_width=True)
    download_csv_button(ct_table, "connection_tracking_features.csv", "Download CT Features")
else:
    st.info("ℹ️ Connection tracking features are available in all datasets.")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)
