import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

st.set_page_config(page_title="Train/Test Comparison - UNSW-NB15", page_icon="🔄", layout="wide")

from utils.loader import load_training_set, load_testing_set
from utils.sidebar import setup_sidebar
from utils.charts import (
    horizontal_bar,
    stacked_bar,
    histogram,
    style_metric_cards,
    download_csv_button,
)
from utils.constants import ATTACK_COLORS, COLORS, FOOTER

style_metric_cards()

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Training Set"

df = setup_sidebar()

with st.spinner("Loading training and testing sets..."):
    train = load_training_set()
    test = load_testing_set()

if train.empty or test.empty:
    st.error("Training or testing set could not be loaded. Check file paths.")
    st.stop()

st.title("🔄 Training / Testing Set Comparison")
st.markdown(
    "This page compares the pre-partitioned training and testing splits to evaluate "
    "distribution consistency. Good train/test splits should maintain similar feature "
    "distributions to ensure reliable model evaluation."
)

train_rows = len(train)
test_rows = len(test)
ratio = train_rows / test_rows if test_rows > 0 else 0
train_attack_pct = train["Label"].mean() * 100
test_attack_pct = test["Label"].mean() * 100
common_cols = len(set(train.columns) & set(test.columns))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Training Rows", f"{train_rows:,}")
k2.metric("Testing Rows", f"{test_rows:,}")
k3.metric("Train/Test Ratio", f"{ratio:.2f}")
k4.metric("Train Attack %", f"{train_attack_pct:.1f}%")
k5.metric("Test Attack %", f"{test_attack_pct:.1f}%")
st.markdown("---")

st.subheader("📊 Label Distribution: Train vs Test")
col1, col2 = st.columns(2)

train_label = train["Label"].value_counts().reset_index()
train_label.columns = ["Label", "Count"]
train_label["Label"] = train_label["Label"].map({0: "Normal", 1: "Attack"})
train_label["Split"] = "Training"

test_label = test["Label"].value_counts().reset_index()
test_label.columns = ["Label", "Count"]
test_label["Label"] = test_label["Label"].map({0: "Normal", 1: "Attack"})
test_label["Split"] = "Testing"

label_compare = pd.concat([train_label, test_label], ignore_index=True)

fig = stacked_bar(
    label_compare, x="Split", y="Count", color="Label",
    title="Label Distribution: Training vs Testing",
    color_map={"Normal": "#3fb950", "Attack": "#f85149"},
    height=400,
    barmode="group",
)
with col1:
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Training set has {train_attack_pct:.1f}% attacks vs "
        f"{test_attack_pct:.1f}% in testing. "
        "Consistent attack ratios between splits ensure fair model evaluation."
    )

with col2:
    label_pct = pd.DataFrame(
        {
            "Split": ["Training", "Testing"],
            "Normal %": [100 - train_attack_pct, 100 - test_attack_pct],
            "Attack %": [train_attack_pct, test_attack_pct],
        }
    )
    st.dataframe(label_pct, width="stretch")

st.markdown("---")

st.subheader("📊 Attack Category Distribution: Train vs Test")
train_cat = train[train["Label"] == 1]["attack_cat"].value_counts().reset_index()
train_cat.columns = ["Attack Category", "Training Count"]

test_cat = test[test["Label"] == 1]["attack_cat"].value_counts().reset_index()
test_cat.columns = ["Attack Category", "Testing Count"]

cat_compare = train_cat.merge(test_cat, on="Attack Category", how="outer").fillna(0)
cat_compare["Training Count"] = cat_compare["Training Count"].astype(int)
cat_compare["Testing Count"] = cat_compare["Testing Count"].astype(int)

cat_melted = cat_compare.melt(
    id_vars=["Attack Category"],
    value_vars=["Training Count", "Testing Count"],
    var_name="Split",
    value_name="Count",
)

fig = stacked_bar(
    cat_melted, x="Attack Category", y="Count", color="Split",
    title="Attack Category Distribution: Training vs Testing",
    color_map={"Training Count": "#00d4ff", "Testing Count": "#ff6b35"},
    height=450,
    barmode="group",
)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Attack category proportions should be similar between splits. Large discrepancies "
    "may indicate sampling bias where certain attack types are over/under-represented "
    "in one split, leading to optimistic or pessimistic model performance estimates."
)
download_csv_button(cat_compare, "attack_category_comparison.csv", "Download Cat Comparison")

st.markdown("---")

st.subheader("📈 Feature Distribution Shift Analysis")
common_features = [
    c for c in train.select_dtypes(include=[np.number]).columns
    if c in test.columns and c != "Label"
]
common_features = [c for c in common_features if train[c].nunique() > 2]

selected_feat = st.selectbox(
    "Select a feature to compare distributions",
    sorted(common_features),
)

sample_train = train.sample(n=min(50000, len(train)))
sample_test = test.sample(n=min(50000, len(test)))

sample_train["Split"] = "Training"
sample_test["Split"] = "Testing"
combined = pd.concat([sample_train[[selected_feat, "Split"]], sample_test[[selected_feat, "Split"]]])

fig = histogram(
    combined,
    x=selected_feat,
    color="Split",
    title=f"{selected_feat}: Training vs Testing Distribution",
    color_map={"Training": "#00d4ff", "Testing": "#ff6b35"},
    opacity=0.5,
    log_x=True,
    barmode="overlay",
    height=400,
)
st.plotly_chart(fig, width="stretch")

train_vals = train[selected_feat].dropna().values
test_vals = test[selected_feat].dropna().values
if len(train_vals) > 0 and len(test_vals) > 0:
    try:
        ks_stat, ks_p = ks_2samp(train_vals, test_vals)
        verdict = "✅ Distributions are similar" if ks_p > 0.05 else "⚠️ Distribution shift detected"
        col1, col2, col3 = st.columns(3)
        col1.metric("KS Statistic", f"{ks_stat:.4f}")
        col2.metric("p-value", f"{ks_p:.6e}")
        col3.metric("Verdict", verdict)
        st.caption(
            f"The Kolmogorov-Smirnov test checks if {selected_feat} comes from the same distribution "
            f"in both splits. {'No significant shift detected.' if ks_p > 0.05 else 'A significant shift suggests the split may not be perfectly representative.'}"
        )
    except Exception:
        st.error("Could not compute KS test.")

st.markdown("---")

st.subheader("⚖️ Class Balance Comparison")
balance = cat_compare.copy()
balance["Train %"] = (balance["Training Count"] / train_rows * 100).round(2)
balance["Test %"] = (balance["Testing Count"] / test_rows * 100).round(2)
balance["Difference %"] = (balance["Train %"] - balance["Test %"]).round(2)

st.dataframe(balance, width="stretch")
download_csv_button(balance, "class_balance_comparison.csv", "Download Balance Comparison")

st.caption(
    "The difference column shows the percentage point disparity between training and testing splits. "
    "Small differences (< 1%) indicate well-stratified splits. Large differences may require "
    "stratified sampling or re-weighting during model training."
)

st.markdown("---")

st.subheader("🔍 Protocol & Service Coverage")
st.markdown("### Checking if all test-set categories appear in training set...")

for col in ["proto", "service", "state"]:
    if col in train.columns and col in test.columns:
        train_vals = set(train[col].dropna().unique())
        test_vals = set(test[col].dropna().unique())
        unseen = test_vals - train_vals

        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric(f"{col}: Train unique", len(train_vals))
        with c2:
            st.metric(f"{col}: Test unique", len(test_vals))

        if unseen:
            st.warning(
                f"⚠️ {len(unseen)} {col} value(s) in testing set not seen in training: "
                f"{', '.join(str(x) for x in sorted(unseen)[:10])}"
                f"{'...' if len(unseen) > 10 else ''}"
            )
        else:
            st.success(f"✅ All {col} values in testing set are present in training set.")

st.caption(
    "If unseen categories appear in the test set, the model may struggle to generalize "
    "to those values. This is especially important for categorical features like protocol "
    "and service, where unseen values may represent novel attack vectors."
)

with st.expander("📋 Full Coverage Table"):
    for col in ["proto", "service", "state"]:
        if col in train.columns and col in test.columns:
            coverage = (
                train[col]
                .value_counts()
                .to_frame("Train Count")
                .join(test[col].value_counts().to_frame("Test Count"), how="outer")
                .fillna(0)
                .astype(int)
            )
            coverage["In Train"] = coverage["Train Count"] > 0
            coverage["In Test"] = coverage["Test Count"] > 0
            st.markdown(f"**{col} Coverage**")
            st.dataframe(coverage, width="stretch")

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>{FOOTER}</div>", unsafe_allow_html=True)
