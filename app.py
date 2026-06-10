import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="UNSW-NB15 - NIDS Data Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.loader import load_dataset
from utils.sidebar import setup_sidebar
from utils.charts import (
    donut_chart,
    style_metric_cards,
    apply_template,
    download_csv_button,
)
from utils.constants import ATTACK_CATEGORIES, FOOTER, COLORS

style_metric_cards()

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 1.5rem; }}
    .hero-title {{
        font-size: 2.5rem; font-weight: 800; color: #e6edf3;
        margin-bottom: 0.5rem; line-height: 1.2;
    }}
    .hero-subtitle {{
        font-size: 1.1rem; color: #8b949e; margin-bottom: 2rem;
        line-height: 1.6;
    }}
    .nav-card {{
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 1.5rem 1rem; text-align: center;
        cursor: pointer; transition: border-color 0.2s;
        height: 100%;
    }}
    .nav-card:hover {{ border-color: #00d4ff; }}
    .nav-card-title {{ font-size: 1.1rem; font-weight: 700; color: #e6edf3; }}
    .nav-card-desc {{ font-size: 0.85rem; color: #8b949e; margin-top: 0.4rem; }}
    .footer {{ text-align: center; color: #8b949e; font-size: 0.8rem;
              padding: 2rem 0 0 0; border-top: 1px solid #21262d; margin-top: 3rem; }}
    .section-header {{ font-size: 1.4rem; font-weight: 700; color: #00d4ff;
                      margin: 2rem 0 1rem 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">🛡️ UNSW-NB15 Network Intrusion Detection</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">'
    "A comprehensive data analysis report on the UNSW-NB15 benchmark dataset for "
    "Network Intrusion Detection Systems (NIDS). This interactive dashboard explores "
    "network traffic patterns, attack behaviours, and feature characteristics across "
    "over 2.5 million records of real modern network traffic mixed with synthetic attacks."
    "</div>",
    unsafe_allow_html=True,
)

df = setup_sidebar()

if df.empty:
    st.error("Dataset not found. Please check the data files are in the correct location.")
    st.stop()

total_records = len(df)
attack_count = int(df["Label"].sum())
normal_count = total_records - attack_count
attack_ratio = attack_count / total_records * 100 if total_records else 0
unique_src_ips = df["srcip"].nunique() if "srcip" in df.columns else "N/A"
unique_dst_ips = df["dstip"].nunique() if "dstip" in df.columns else "N/A"
attack_types = df["attack_cat"].nunique() - (1 if "Normal" in df["attack_cat"].values else 0)

k1, k2, k3, k4, k5 = st.columns(5)
metrics = [
    (f"{total_records:,}", "", "Total Records"),
    (f"{attack_count:,}", f"{attack_ratio:.1f}%", "Attack Records"),
    (f"{normal_count:,}", f"{100-attack_ratio:.1f}%", "Normal Records"),
    (str(unique_src_ips), "", "Unique Source IPs") if unique_src_ips != "N/A" else ("N/A", "", "Unique Source IPs"),
    (f"{attack_types}", "", "Attack Categories"),
]
for col, (val, delta, label) in zip([k1, k2, k3, k4, k5], metrics):
    col.metric(label=label, value=val, delta=delta)

st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    labels = ["Normal", "Attack"]
    values = [normal_count, attack_count]
    fig = donut_chart(
        labels=labels,
        values=values,
        title="Normal vs Attack Distribution",
        colors=["#3fb950", "#f85149"],
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The dataset contains a mix of normal and attack traffic. "
        f"Attacks comprise {attack_ratio:.1f}% of all records, providing a "
        "moderately imbalanced classification problem typical in intrusion detection."
    )

with col2:
    st.markdown('<div class="section-header">Quick Navigation</div>', unsafe_allow_html=True)
    nav_items = [
        ("📊", "Dataset Overview", "Structure, types, missing values & samples"),
        ("⚔️", "Attack Analysis", "Category distributions, deep dives & patterns"),
        ("🌐", "Traffic Analysis", "Protocols, services, states & volume"),
        ("📡", "Network Analysis", "IPs, ports, TTLs & duration patterns"),
        ("🔬", "Feature Analysis", "Distributions, correlations & TCP features"),
        ("📈", "Statistical Analysis", "Hypothesis tests, importance & outliers"),
        ("🔄", "Train/Test Comparison", "Split quality, shifts & balance"),
    ]
    pages = [
        "01_overview", "02_attack_analysis", "03_traffic_analysis",
        "04_network_analysis", "05_feature_analysis", "06_statistical_analysis",
        "07_train_test_comparison",
    ]
    for i in range(0, len(nav_items), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(nav_items):
                emoji, title, desc = nav_items[idx]
                with cols[j]:
                    st.markdown(
                        f'<a href="/{pages[idx]}" target="_self" style="text-decoration:none;">'
                        f'<div class="nav-card">'
                        f'<div style="font-size:2rem;margin-bottom:0.3rem;">{emoji}</div>'
                        f'<div class="nav-card-title">{title}</div>'
                        f'<div class="nav-card-desc">{desc}</div>'
                        f'</div></a>',
                        unsafe_allow_html=True,
                    )

st.markdown("---")

with st.expander("ℹ️ About the Dataset — UNSW-NB15"):
    st.markdown(
        """
    **What is UNSW-NB15?**
    The UNSW-NB15 dataset was created by the Australian Centre for Cyber Security (ACCS)
    at UNSW Canberra. It contains a hybrid of real modern network traffic and synthetic
    attack behaviours across 9 attack categories.

    **Key Characteristics:**
    - **2,540,044** records in the full raw dataset
    - **9 attack categories**: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
      Reconnaissance, Shellcode, Worms
    - **49 features** including flow, basic, content, time, and additional generated features
    - Pre-partitioned into **training (175,341)** and **testing (82,332)** sets

    **Why it matters:**
    Unlike older datasets (KDD'99, NSL-KDD), UNSW-NB15 reflects modern network traffic
    patterns and sophisticated attack techniques. It has become a standard benchmark for
    evaluating NIDS machine learning models.

    **Data Sources:**
    - IXIA PerfectStorm tool for traffic generation
    - 12 different network protocols
    - 100 GB of raw traffic captured in PCAP files, then processed into CSV format
    """,
        unsafe_allow_html=True,
    )

    df_info = pd.DataFrame(
        {
            "Property": [
                "Total Records",
                "Total Features",
                "Attack Categories",
                "Normal Records",
                "Attack Records",
                "Attack Ratio",
                "Memory Usage",
            ],
            "Value": [
                f"{total_records:,}",
                df.shape[1],
                9,
                f"{normal_count:,}",
                f"{attack_count:,}",
                f"{attack_ratio:.1f}%",
                f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
            ],
        }
    )
    download_csv_button(df_info, "dataset_info.csv", label="Download Dataset Info")

st.markdown(
    f'<div class="footer">{FOOTER}</div>',
    unsafe_allow_html=True,
)
