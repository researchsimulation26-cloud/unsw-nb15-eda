import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Traffic Analysis - UNSW-NB15", page_icon="🌐", layout="wide")

from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    horizontal_bar,
    stacked_bar,
    histogram,
    heatmap,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import ATTACK_CATEGORIES, ATTACK_COLORS, COLORS, FOOTER

style_metric_cards()

df = setup_sidebar()

if df.empty:
    st.error("Dataset not loaded.")
    st.stop()

st.title("🌐 Traffic Analysis")
st.markdown(
    "This page examines the protocol, service, and state characteristics of network traffic, "
    "along with traffic volume patterns across attack categories."
)

total_flows = len(df)
unique_protos = df["proto"].nunique()
unique_services = df["service"].nunique()
unique_states = df["state"].nunique()
top_proto = df["proto"].mode().iloc[0] if "proto" in df.columns else "N/A"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Flows", f"{total_flows:,}")
k2.metric("Unique Protocols", unique_protos)
k3.metric("Unique Services", unique_services)
k4.metric("Unique States", unique_states)
k5.metric("Most Common Proto", top_proto)
st.markdown("---")

st.subheader("📡 Protocol Distribution")
top_n = st.slider("Number of protocols to show", 5, 20, 15, key="proto_topn")
proto_counts = df["proto"].value_counts().head(top_n).reset_index()
proto_counts.columns = ["Protocol", "Count"]

col1, col2 = st.columns([1, 1])
with col1:
    fig = horizontal_bar(
        proto_counts, y="Protocol", x="Count",
        title=f"Top {top_n} Protocols by Count",
        color="#00d4ff", sort_values=True, height=400,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    attacks = df[df["Label"] == 1]
    proto_cat = (
        attacks.groupby(["proto", "attack_cat"])
        .size()
        .reset_index(name="count")
    )
    top_protos = attacks["proto"].value_counts().head(10).index
    proto_cat = proto_cat[proto_cat["proto"].isin(top_protos)]
    fig = stacked_bar(
        proto_cat, x="proto", y="count", color="attack_cat",
        title="Protocol × Attack Category (Attacks Only)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

st.caption(
    f"The top protocols reveal the dominant communication types in the dataset. "
    f"TCP and UDP typically dominate. The stacked bar shows which attack categories "
    f"are most associated with each protocol — for example, certain exploits may "
    f"concentrate on HTTP (TCP port 80) while scans may use UDP."
)

st.markdown("---")

st.subheader("🔌 Service Distribution")
service_counts = df["service"].value_counts().head(top_n).reset_index()
service_counts.columns = ["Service", "Count"]

col1, col2 = st.columns([1, 1])
with col1:
    fig = horizontal_bar(
        service_counts, y="Service", x="Count",
        title=f"Top {top_n} Services by Count",
        color="#ff6b35", sort_values=True, height=400,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    attacks = df[df["Label"] == 1]
    svc_cat = (
        attacks.groupby(["service", "attack_cat"])
        .size()
        .reset_index(name="count")
    )
    top_svcs = attacks["service"].value_counts().head(10).index
    svc_cat = svc_cat[svc_cat["service"].isin(top_svcs)]
    fig = stacked_bar(
        svc_cat, x="service", y="count", color="attack_cat",
        title="Service × Attack Category (Attacks Only)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Services like DNS, HTTP, and FTP are common attack vectors. The stacked chart "
    "highlights which services are most frequently targeted or exploited by each attack "
    "category. For example, HTTP is a common vector for web-based exploits."
)

st.markdown("---")

st.subheader("🔗 Connection State Analysis")
st.markdown("**What do connection states mean?**")
st.info(
    "- **CON**: Connection established\n"
    "- **FIN**: Normal connection termination\n"
    "- **INT**: Connection interrupted\n"
    "- **RST**: Connection reset\n"
    "- **REQ**: Request state\n"
    "- **ECO/ECR**: Echo request/reply (often ICMP)\n"
    "- **CLO**: Connection closed\n"
    "- **ACC**: Connection accepted\n"
    "- **URH/URN**: Urgent header/pointer states\n"
    "- **PAR**: Partial connection\n"
    "- **TST**: Test state"
)

state_counts = df["state"].value_counts().head(top_n).reset_index()
state_counts.columns = ["State", "Count"]

col1, col2 = st.columns([1, 1])
with col1:
    fig = horizontal_bar(
        state_counts, y="State", x="Count",
        title=f"Top {top_n} Connection States",
        color="#79c0ff", sort_values=True, height=400,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    attacks = df[df["Label"] == 1]
    state_cat = (
        attacks.groupby(["state", "attack_cat"])
        .size()
        .reset_index(name="count")
    )
    top_states = attacks["state"].value_counts().head(10).index
    state_cat = state_cat[state_cat["state"].isin(top_states)]
    fig = stacked_bar(
        state_cat, x="state", y="count", color="attack_cat",
        title="State × Attack Category (Attacks Only)",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")

st.caption(
    "Connection states like RST (reset) and INT (interrupted) are more prevalent in "
    "attack traffic due to aborted connections, port scans, and rejected connections. "
    "FIN states are more common in legitimate connections with proper termination."
)

st.markdown("---")

st.subheader("🌡️ Traffic Volume Heatmap")
if len(df[df["Label"] == 1]) > 0:
    attacks = df[df["Label"] == 1]
    pivot_bytes = attacks.pivot_table(
        index="proto", columns="attack_cat", values="sbytes", aggfunc="mean"
    ).fillna(0)
    pivot_bytes = pivot_bytes.loc[
        pivot_bytes.sum(axis=1).sort_values(ascending=False).head(15).index
    ]

    fig = heatmap(
        pivot_bytes,
        title="Mean Source Bytes: Protocol × Attack Category",
        colorscale="Viridis",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "This heatmap shows the average source bytes per flow for each protocol and "
        "attack category combination. Brighter cells indicate higher byte volumes. "
        "Protocols that carry large payloads for specific attack types appear as hotspots."
    )

    pivot_dbytes = attacks.pivot_table(
        index="proto", columns="attack_cat", values="dbytes", aggfunc="mean"
    ).fillna(0)
    pivot_dbytes = pivot_dbytes.loc[
        pivot_dbytes.sum(axis=1).sort_values(ascending=False).head(15).index
    ]
    fig = heatmap(
        pivot_dbytes,
        title="Mean Destination Bytes: Protocol × Attack Category",
        colorscale="Viridis",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Destination byte heatmap complements the source view. Asymmetry between "
        "source and destination bytes can reveal attack characteristics — e.g., "
        "data exfiltration shows high dst bytes, while command injection shows low dst bytes."
    )

st.markdown("---")

st.subheader("📦 Bytes & Packets Distributions")
sample_df = df.sample(n=min(200000, len(df))) if len(df) > 200000 else df
sample_df["label_name"] = sample_df["Label"].map({0: "Normal", 1: "Attack"})

dist_features = [
    ("sbytes", "Source Bytes"),
    ("dbytes", "Destination Bytes"),
    ("Spkts", "Source Packets"),
    ("Dpkts", "Destination Packets"),
]

for i in range(0, 4, 2):
    cols = st.columns(2)
    for j in range(2):
        feat, feat_label = dist_features[i + j]
        fig = histogram(
            sample_df, x=feat, color="label_name",
            title=f"{feat_label} Distribution (Log Scale)",
            log_x=True,
            height=350,
        )
        with cols[j]:
            st.plotly_chart(fig, width="stretch")

if len(df) > 200000:
    st.info("📌 Charts based on 200k sample for performance")

st.caption(
    "Log scale is used because network traffic features follow a power-law distribution — "
    "most flows are small (e.g., DNS queries) while a few are very large (e.g., file transfers). "
    "Attack traffic often shows distinct distribution patterns, such as bimodal behaviour "
    "where small packets represent scans and large packets represent data exfiltration."
)

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)
