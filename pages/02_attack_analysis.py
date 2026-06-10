import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Attack Analysis - UNSW-NB15", page_icon="⚔️", layout="wide")

import plotly.graph_objects as go
from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    horizontal_bar,
    stacked_bar,
    donut_chart,
    violin_plot,
    line_chart,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import ATTACK_CATEGORIES, ATTACK_COLORS, COLORS, FOOTER

style_metric_cards()

df = setup_sidebar()

if df.empty:
    st.error("Dataset not loaded.")
    st.stop()

attacks = df[df["Label"] == 1].copy()
st.title("⚔️ Attack Analysis")
st.markdown(
    "This page focuses exclusively on attack traffic — examining category distributions, "
    "temporal patterns, and feature differences between attack and normal behaviour."
)

total_attacks = len(attacks)
attack_pct = total_attacks / len(df) * 100
avg_dur = attacks["dur"].mean()
avg_bytes = attacks["sbytes"].mean()
avg_pkts = attacks["Spkts"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Attack Records", f"{total_attacks:,}")
k2.metric("% of Dataset", f"{attack_pct:.1f}%")
k3.metric("Avg Duration (s)", f"{avg_dur:.4f}")
k4.metric("Avg Src Bytes", f"{avg_bytes:,.0f}")
k5.metric("Avg Src Packets", f"{avg_pkts:,.2f}")
st.markdown("---")

cat_counts = attacks["attack_cat"].value_counts().reset_index()
cat_counts.columns = ["Attack Category", "Count"]
cat_counts["% of Attacks"] = (cat_counts["Count"] / total_attacks * 100).round(2)
cat_counts["% of Total"] = (cat_counts["Count"] / len(df) * 100).round(2)

st.subheader("📊 Attack Category Distribution")
col1, col2 = st.columns([1, 1])

with col1:
    fig = horizontal_bar(
        cat_counts, y="Attack Category", x="Count",
        title="Attack Categories by Count",
        color="#ff6b35", sort_values=True, height=400,
    )
    fig.update_traces(marker_color=[ATTACK_COLORS.get(c, "#ff6b35") for c in cat_counts["Attack Category"]])
    st.plotly_chart(fig, width="stretch")

with col2:
    cat_colors = [ATTACK_COLORS.get(c, "#ff6b35") for c in cat_counts["Attack Category"]]
    fig = donut_chart(
        labels=cat_counts["Attack Category"].tolist(),
        values=cat_counts["Count"].tolist(),
        title="Attack Category Proportions",
        colors=cat_colors,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Generic and Exploits are typically the most prevalent categories in UNSW-NB15, "
    "reflecting the prevalence of broad-scope attacks in modern network traffic. "
    "The distribution reveals which attack types are most commonly represented in the dataset."
)

st.markdown("---")

st.subheader("📋 Attack Summary Table")
st.dataframe(cat_counts, width="stretch")
download_csv_button(cat_counts, "attack_distribution.csv", "Download Attack Distribution")

st.markdown("---")

st.subheader("🔍 Attack Category Deep Dive")
selected_cat = st.selectbox("Select an Attack Category", sorted(attacks["attack_cat"].unique()))
cat_df = attacks[attacks["attack_cat"] == selected_cat]
other_df = attacks[attacks["attack_cat"] != selected_cat]

cat_count = len(cat_df)
cat_pct_attacks = cat_count / total_attacks * 100
cat_pct_total = cat_count / len(df) * 100
cat_avg_dur = cat_df["dur"].mean()
cat_avg_sbytes = cat_df["sbytes"].mean()
cat_avg_spkts = cat_df["Spkts"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Count", f"{cat_count:,}")
c2.metric("% of Attacks", f"{cat_pct_attacks:.1f}%")
c3.metric("Avg Duration", f"{cat_avg_dur:.4f}s")
c4.metric("Avg Src Bytes", f"{cat_avg_sbytes:,.0f}")
c5.metric("Avg Src Packets", f"{cat_avg_spkts:,.0f}")

radar_features = ["dur", "sbytes", "dbytes", "Spkts", "Dpkts", "Sload", "Dload"]
cat_means = cat_df[radar_features].mean()
other_means = other_df[radar_features].mean() if len(other_df) > 0 else 0

cat_norm = (cat_means - cat_means.min()) / (cat_means.max() - cat_means.min() + 1e-10)
other_norm = (other_means - cat_means.min()) / (cat_means.max() - cat_means.min() + 1e-10)

fig = go.Figure()
fig.add_trace(
    go.Scatterpolar(
        r=cat_norm.values,
        theta=radar_features,
        fill="toself",
        name=selected_cat,
        line=dict(color="#00d4ff", width=2),
    )
)
fig.add_trace(
    go.Scatterpolar(
        r=other_norm.values,
        theta=radar_features,
        fill="toself",
        name=f"Other Attacks",
        line=dict(color="#8b949e", width=2),
    )
)
fig.update_layout(
    title=dict(text=f"Feature Profile: {selected_cat} vs Other Attacks", font=dict(size=16, color="#e6edf3"), x=0.5),
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], gridcolor="#21262d", tickfont=dict(color="#8b949e", size=10)),
        bgcolor="rgba(0,0,0,0)",
        angularaxis=dict(gridcolor="#21262d", tickfont=dict(color="#e6edf3", size=11)),
    ),
    height=500,
    margin=dict(l=40, r=40, t=50, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3"),
    legend=dict(font=dict(color="#e6edf3")),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"The radar chart compares {selected_cat}'s feature profile against all other attack categories. "
    "Features are normalized to [0,1] for comparison. Distinct shapes indicate different attack "
    "behaviours — e.g., DoS attacks typically show high packet counts but short durations."
)

st.markdown("---")

st.subheader("⏱️ Temporal Patterns")
if "Stime" in df.columns:
    df_time = attacks.copy()
    df_time["datetime"] = pd.to_datetime(df_time["Stime"], unit="s", errors="coerce")
    df_time = df_time.dropna(subset=["datetime"])
    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["day"] = df_time["datetime"].dt.date

    hourly = df_time.groupby("hour").size().reset_index(name="count")
    fig = line_chart(hourly, x="hour", y="count", title="Attacks by Hour of Day", color="#f85149")
    fig.update_layout(xaxis=dict(dtick=2))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Temporal clustering of attacks may indicate coordinated attack campaigns or "
        "automated scanning activity. Peaks at specific hours could reveal attacker "
        "timezone preferences or scheduled attack tool behaviour."
    )

    hourly_cat = df_time.groupby(["hour", "attack_cat"]).size().reset_index(name="count")
    fig = line_chart(
        hourly_cat, x="hour", y="count", color="attack_cat",
        title="Attack Categories Over Time (Hourly)",
        color_map=ATTACK_COLORS,
    )
    fig.update_layout(xaxis=dict(dtick=2))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Different attack categories exhibit distinct temporal signatures. For example, "
        "automated scans (Reconnaissance) may show uniform distribution while targeted "
        "exploits may cluster within specific time windows."
    )
else:
    st.info("ℹ️ Temporal analysis requires the Full Raw Data dataset (contains Stime column). "
            "Switch to Full Raw Data in the sidebar.")

st.markdown("---")

st.subheader("📦 Attack vs Normal Feature Comparison")
violin_features = st.multiselect(
    "Select features for violin plots",
    ["dur", "sbytes", "dbytes", "Spkts", "Dpkts"],
    default=["dur", "sbytes", "dbytes", "Spkts", "Dpkts"],
)

df_violin = df.copy()
df_violin["label_name"] = df_violin["Label"].map({0: "Normal", 1: "Attack"})

ncols = min(2, len(violin_features))
for i in range(0, len(violin_features), ncols):
    cols = st.columns(ncols)
    for j, feat in enumerate(violin_features[i : i + ncols]):
        use_log = feat in ["sbytes", "dbytes", "Spkts", "Dpkts"]
        sample = df_violin.sample(n=min(100000, len(df_violin))) if len(df_violin) > 200000 else df_violin
        fig = violin_plot(
            sample, x="label_name", y=feat,
            title=f"{feat}: Normal vs Attack",
            log_y=use_log,
            height=400,
        )
        with cols[j]:
            st.plotly_chart(fig, width="stretch")

st.caption(
    "Violin plots reveal distributional differences between normal and attack traffic. "
    "Attack traffic often exhibits distinct byte count and packet count distributions — "
    "for example, DoS floods generate many small packets while exploit payloads are larger. "
    "Log scale is used for skewed features to better visualize distribution shapes."
)

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)

import plotly.graph_objects as go
