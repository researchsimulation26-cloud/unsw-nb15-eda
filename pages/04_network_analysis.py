import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Network Analysis - UNSW-NB15", page_icon="📡", layout="wide")

from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    horizontal_bar,
    stacked_bar,
    box_plot,
    colored_bar,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import ATTACK_COLORS, COLORS, FOOTER

style_metric_cards()


def _add_port_annotations(fig, ports):
    known_ports = {80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP",
                   53: "DNS", 25: "SMTP", 110: "POP3", 23: "Telnet",
                   3389: "RDP", 8080: "HTTP-Alt"}
    for i, p in enumerate(ports):
        p_int = int(p)
        if p_int in known_ports:
            fig.add_annotation(
                x=0, y=i,
                text=f" ({known_ports[p_int]})",
                showarrow=False,
                font=dict(size=10, color="#8b949e"),
                xshift=5,
            )
    return fig


df = setup_sidebar()

if df.empty:
    st.error("Dataset not loaded.")
    st.stop()

has_ip = "srcip" in df.columns and "dstip" in df.columns
has_port = "sport" in df.columns and "dsport" in df.columns

st.title("📡 Network Analysis")
st.markdown(
    "This page explores network-level characteristics: IP addresses, port usage, TTL values "
    "for OS fingerprinting, and connection duration patterns."
)

total_flows = len(df)
if has_ip:
    unique_src = df["srcip"].nunique()
    unique_dst = df["dstip"].nunique()
    overlap = len(set(df["srcip"].unique()) & set(df["dstip"].unique()))
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Flows", f"{total_flows:,}")
    k2.metric("Unique Source IPs", f"{unique_src:,}")
    k3.metric("Unique Dest IPs", f"{unique_dst:,}")
    k4.metric("IP Overlap", f"{overlap:,}")
    k5.metric("Attack Ratio", f'{df["Label"].mean()*100:.1f}%')
    st.info(
        "Note: IP addresses in the UNSW-NB15 dataset are anonymized/synthetic. "
        "They represent network topology rather than real internet addresses."
    )
else:
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Flows", f"{total_flows:,}")
    k2.metric("Attack Ratio", f'{df["Label"].mean()*100:.1f}%')
    k3.metric("Has IP Data", "No")
    st.warning("⚠️ IP address data is only available in the Full Raw Data dataset. "
               "Switch to Full Raw Data in the sidebar for full network analysis.")

st.markdown("---")

if has_ip:
    st.subheader("📍 Source IP Analysis")
    src_counts = df["srcip"].value_counts().head(15).reset_index()
    src_counts.columns = ["srcip", "Count"]

    top_attack_per_ip = (
        df[df["Label"] == 1]
        .groupby("srcip")["attack_cat"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
        .reset_index()
    )
    src_with_attack = src_counts.merge(top_attack_per_ip, on="srcip", how="left")
    src_with_attack["attack_cat"] = src_with_attack["attack_cat"].fillna("Normal")

    fig = colored_bar(
        src_with_attack, y="srcip", x="Count", color_col="attack_cat",
        title="Top 15 Most Active Source IPs",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The most active source IPs are shown, colored by their dominant attack type. "
        "IPs with high activity and a single dominant attack category may represent "
        "dedicated attack sources or compromised hosts."
    )

    st.subheader("📍 Destination IP Analysis")
    dst_counts = df["dstip"].value_counts().head(15).reset_index()
    dst_counts.columns = ["Destination IP", "Count"]

    fig = horizontal_bar(
        dst_counts, y="Destination IP", x="Count",
        title="Top 15 Most Targeted Destination IPs",
        color="#f85149", sort_values=True, height=450,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Destination IPs receiving the most traffic may represent critical servers or "
        "services under attack. Concentrated targeting of specific IPs suggests focused "
        "attack campaigns or popular service endpoints."
    )

st.markdown("---")

if has_port:
    st.subheader("🔌 Port Analysis")

    df["sport_n"] = pd.to_numeric(df["sport"], errors="coerce")
    df["dsport_n"] = pd.to_numeric(df["dsport"], errors="coerce")

    col1, col2 = st.columns(2)
    with col1:
        src_ports = df["sport_n"].value_counts().head(20).reset_index()
        src_ports.columns = ["Port", "Count"]
        fig = horizontal_bar(
            src_ports, y="Port", x="Count",
            title="Top 20 Source Ports",
            color="#00d4ff", sort_values=True, height=450,
        )
        fig = _add_port_annotations(fig, src_ports["Port"].tolist())
        st.plotly_chart(fig, width="stretch")

    with col2:
        dst_ports = df["dsport_n"].value_counts().head(20).reset_index()
        dst_ports.columns = ["Port", "Count"]
        fig = horizontal_bar(
            dst_ports, y="Port", x="Count",
            title="Top 20 Destination Ports",
            color="#ff6b35", sort_values=True, height=450,
        )
        fig = _add_port_annotations(fig, dst_ports["Port"].tolist())
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "Well-known ports: 80 (HTTP), 443 (HTTPS), 22 (SSH), 21 (FTP), 53 (DNS), "
        "25 (SMTP), 110 (POP3). Anomalies such as non-standard ports carrying significant "
        "traffic may indicate covert channels or unusual service configurations."
    )

    attacks = df[df["Label"] == 1]
    if len(attacks) > 0:
        top_dstports = attacks["dsport_n"].value_counts().head(10).index
        port_cat = (
            attacks[attacks["dsport_n"].isin(top_dstports)]
            .groupby(["dsport_n", "attack_cat"])
            .size()
            .reset_index(name="count")
        )
        port_cat["dsport_n"] = port_cat["dsport_n"].astype(int)
        fig = stacked_bar(
            port_cat, x="dsport_n", y="count", color="attack_cat",
            title="Top 10 Destination Ports × Attack Category",
            height=400,
        )
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Certain ports are more frequently targeted by specific attack types. "
            "For example, port 80 (HTTP) is a common vector for web exploits, "
            "while port 22 (SSH) may be targeted by brute-force and reconnaissance attacks."
        )
else:
    st.info("ℹ️ Port analysis requires the Full Raw Data dataset.")

st.markdown("---")

st.subheader("🖥️ TTL Analysis — OS Fingerprinting")
st.info(
    "TTL (Time To Live) values can help identify the operating system of source/destination hosts:\n"
    "- **TTL 64** → Linux / Unix / macOS\n"
    "- **TTL 128** → Windows\n"
    "- **TTL 255** → Cisco / Network devices\n\n"
    "Attack traffic may show different TTL distributions if attackers use different OSes "
    "than typical network users, or if they manipulate TTL values to evade detection."
)

col1, col2 = st.columns(2)
with col1:
    sttls = df["sttl"].value_counts().head(15).reset_index()
    sttls.columns = ["TTL", "Count"]
    fig = horizontal_bar(
        sttls, y="TTL", x="Count",
        title="Source TTL Distribution",
        color="#79c0ff", sort_values=True, height=400,
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    dttls = df["dttl"].value_counts().head(15).reset_index()
    dttls.columns = ["TTL", "Count"]
    fig = horizontal_bar(
        dttls, y="TTL", x="Count",
        title="Destination TTL Distribution",
        color="#a5d6ff", sort_values=True, height=400,
    )
    st.plotly_chart(fig, width="stretch")

attacks_ttl = df[df["Label"] == 1]
if len(attacks_ttl) > 0:
    sttls_top = attacks_ttl["sttl"].value_counts().head(10).index
    sttl_cat = (
        attacks_ttl[attacks_ttl["sttl"].isin(sttls_top)]
        .groupby(["sttl", "attack_cat"])
        .size()
        .reset_index(name="count")
    )
    fig = stacked_bar(
        sttl_cat, x="sttl", y="count", color="attack_cat",
        title="Source TTL × Attack Category",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "TTL values associated with specific attack categories can reveal attacker infrastructure. "
        "For instance, a concentration of TTL=128 (Windows) among certain attack types may indicate "
        "the predominant OS used by those attackers."
    )

st.markdown("---")

st.subheader("⏳ Connection Duration Analysis")
dur_df = df[df["dur"] > 0].copy()
dur_df["label_name"] = dur_df["Label"].map({0: "Normal", 1: "Attack"})

sample_dur = dur_df.sample(n=min(100000, len(dur_df))) if len(dur_df) > 100000 else dur_df

col1, col2 = st.columns(2)
with col1:
    fig = box_plot(
        sample_dur, x="attack_cat", y="dur",
        title="Duration by Attack Category (Log Scale, No Outliers)",
        log_y=True,
        height=500,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "DoS attacks typically have very short durations due to their flood-based nature — "
        "they send many packets quickly without establishing connections. Exploits and "
        "Backdoors may show longer durations as they maintain persistent connections."
    )

with col2:
    median_dur = dur_df.groupby("attack_cat")["dur"].median().reset_index()
    median_dur = median_dur.sort_values("dur", ascending=True)
    cat_colors = [ATTACK_COLORS.get(c, "#ff6b35") for c in median_dur["attack_cat"]]
    fig = horizontal_bar(
        median_dur, y="attack_cat", x="dur",
        title="Median Duration per Attack Category",
        color="#00d4ff", sort_values=True, height=450,
    )
    fig.update_traces(marker_color=cat_colors)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Median duration varies significantly across attack categories. Reconnaissance "
        "and scanning activities are often brief, while exploits and backdoors may "
        "require longer connection times for payload delivery or command execution."
    )

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)
