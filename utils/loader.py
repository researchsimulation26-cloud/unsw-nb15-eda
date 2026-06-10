import pandas as pd
import streamlit as st
from pathlib import Path

from .constants import RAW_COLUMN_NAMES, TRAIN_TEST_COLS_MAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_HF_REPO = "research-simulation26/unsw-nb15-parquet"
_KAGGLE_RAW = "harshwardhanbhangale/unsw-complete-dataset"
_KAGGLE_TRAIN_TEST = "dhoogla/unswnb15"


def _fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename = {k: v for k, v in TRAIN_TEST_COLS_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].fillna("Normal")
    if "ct_flw_http_mthd" in df.columns:
        df["ct_flw_http_mthd"] = df["ct_flw_http_mthd"].fillna(0)
    if "is_ftp_login" in df.columns:
        df["is_ftp_login"] = df["is_ftp_login"].fillna(0)
    return df


def _download_from_hf(filename: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=_HF_REPO, filename=filename, repo_type="dataset")
    return pd.read_parquet(path)


def _download_kaggle(kaggle_path: str) -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download(kaggle_path))


def _find_local_parquet(name: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "raw_data" / name,
        PROJECT_ROOT / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_local_raw(part_num: int) -> Path | None:
    candidates = [
        PROJECT_ROOT / "raw_data" / f"UNSW-NB15_{part_num}.csv",
        PROJECT_ROOT / f"UNSW-NB15_{part_num}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_local_csv(filename: str) -> Path | None:
    candidates = [
        PROJECT_ROOT / "raw_data" / "Training and Testing Sets" / filename,
        PROJECT_ROOT / "Training and Testing Sets" / filename,
        PROJECT_ROOT / "raw_data" / filename,
        PROJECT_ROOT / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner="Loading raw data...")
def load_raw_data() -> pd.DataFrame:
    try:
        local_pq = _find_local_parquet("raw_data.parquet")
        if local_pq:
            return _fix_column_names(pd.read_parquet(local_pq))

        return _fix_column_names(_download_from_hf("raw_data.parquet"))
    except Exception:
        pass

    try:
        parts = []
        for i in range(1, 5):
            local = _find_local_raw(i)
            if local:
                df_part = pd.read_csv(local, header=None, low_memory=False, encoding="latin1")
                parts.append(df_part)

        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.columns = RAW_COLUMN_NAMES
            return _fix_column_names(df)

        download_path = _download_kaggle(_KAGGLE_RAW)
        for i in range(1, 5):
            f = download_path / f"UNSW-NB15_{i}.csv"
            if f.exists():
                df_part = pd.read_csv(f, header=None, low_memory=False, encoding="latin1")
                parts.append(df_part)

        if not parts:
            raise FileNotFoundError("Raw data not found locally or on Kaggle")

        df = pd.concat(parts, ignore_index=True)
        df.columns = RAW_COLUMN_NAMES
        return _fix_column_names(df)
    except Exception as e:
        st.error(f"Failed to load raw data: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading training set...")
def load_training_set() -> pd.DataFrame:
    try:
        local_pq = _find_local_parquet("training_set.parquet")
        if local_pq:
            df = pd.read_parquet(local_pq)
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            return _fix_column_names(df)

        df = _download_from_hf("training_set.parquet")
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        return _fix_column_names(df)
    except Exception:
        pass

    try:
        local = _find_local_csv("UNSW_NB15_training-set.csv")
        if local:
            df = pd.read_csv(local, low_memory=False)
        else:
            download_path = _download_kaggle(_KAGGLE_TRAIN_TEST)
            pq_path = download_path / "UNSW_NB15_training-set.parquet"
            if pq_path.exists():
                df = pd.read_parquet(pq_path)
            else:
                raise FileNotFoundError("Training set not found")
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        return _fix_column_names(df)
    except Exception as e:
        st.error(f"Failed to load training set: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading testing set...")
def load_testing_set() -> pd.DataFrame:
    try:
        local_pq = _find_local_parquet("testing_set.parquet")
        if local_pq:
            df = pd.read_parquet(local_pq)
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            return _fix_column_names(df)

        df = _download_from_hf("testing_set.parquet")
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        return _fix_column_names(df)
    except Exception:
        pass

    try:
        local = _find_local_csv("UNSW_NB15_testing-set.csv")
        if local:
            df = pd.read_csv(local, low_memory=False)
        else:
            download_path = _download_kaggle(_KAGGLE_TRAIN_TEST)
            pq_path = download_path / "UNSW_NB15_testing-set.parquet"
            if pq_path.exists():
                df = pd.read_parquet(pq_path)
            else:
                raise FileNotFoundError("Testing set not found")
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        return _fix_column_names(df)
    except Exception as e:
        st.error(f"Failed to load testing set: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset(name: str) -> pd.DataFrame:
    if name == "Full Raw Data":
        return load_raw_data()
    elif name == "Training Set":
        return load_training_set()
    elif name == "Testing Set":
        return load_testing_set()
    return load_training_set()
