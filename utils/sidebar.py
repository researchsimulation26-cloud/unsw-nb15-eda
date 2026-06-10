import streamlit as st
import pandas as pd
from .loader import load_dataset
from .constants import ATTACK_CATEGORIES, FOOTER


def setup_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='text-align:center;'>🛡️ UNSW-NB15</h2>"
            "<p style='text-align:center;color:#8b949e;font-size:0.85rem;'>NIDS Data Report</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        ds_name = st.selectbox(
            "Dataset",
            options=["Training Set", "Testing Set", "Full Raw Data"],
            index=0,
            key="dataset_selector",
        )
        st.session_state.dataset_name = ds_name

        with st.spinner("Loading..."):
            df = load_dataset(ds_name)

        if not df.empty:
            st.markdown("---")
            st.markdown("### 🔍 Filters")

            attack_options = sorted(
                df["attack_cat"].dropna().unique()
            )
            selected_attacks = st.multiselect(
                "Attack Categories",
                options=attack_options,
                default=st.session_state.get("selected_attacks", []),
                key="sidebar_attacks",
            )
            st.session_state.selected_attacks = selected_attacks

            if "proto" in df.columns:
                proto_options = sorted(df["proto"].dropna().unique().tolist())
                selected_protos = st.multiselect(
                    "Protocols",
                    options=proto_options,
                    default=st.session_state.get("selected_protocols", []),
                    key="sidebar_protos",
                )
                st.session_state.selected_protocols = selected_protos

            st.markdown("---")
            st.markdown("### 📋 Dataset Info")
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(
                f"- **Rows:** {len(df):,}\n"
                f"- **Columns:** {df.shape[1]}\n"
                f"- **Memory:** {memory_mb:.1f} MB\n"
                f"- **Attack %:** {df['Label'].mean()*100:.1f}%"
            )

        st.markdown("---")
        st.markdown(
            f"<div style='text-align:center;color:#8b949e;font-size:0.7rem;'>{FOOTER}</div>",
            unsafe_allow_html=True,
        )

    selected_attacks = st.session_state.get("selected_attacks", [])
    selected_protos = st.session_state.get("selected_protocols", [])
    if selected_attacks:
        df = df[df["attack_cat"].isin(selected_attacks)]
    if selected_protos and "proto" in df.columns:
        df = df[df["proto"].isin(selected_protos)]
    return df
