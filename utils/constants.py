from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_PATH / "raw_data"

COLORS = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "primary": "#00d4ff",
    "secondary": "#ff6b35",
    "success": "#3fb950",
    "danger": "#f85149",
    "text": "#e6edf3",
    "muted": "#8b949e",
}

ATTACK_COLORS = {
    "Normal": "#3fb950",
    "Fuzzers": "#ff6b35",
    "Analysis": "#d2a8ff",
    "Backdoors": "#f85149",
    "DoS": "#ffd700",
    "Exploits": "#ff7b72",
    "Generic": "#79c0ff",
    "Reconnaissance": "#a5d6ff",
    "Shellcode": "#c9d1d9",
    "Worms": "#ffa657",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#e6edf3"},
        "xaxis": {"gridcolor": "#21262d", "zerolinecolor": "#30363d"},
        "yaxis": {"gridcolor": "#21262d", "zerolinecolor": "#30363d"},
        "hoverlabel": {"bgcolor": "#161b22", "bordercolor": "#30363d"},
    }
}

CATEGORICAL_COLS = ["proto", "state", "service", "attack_cat"]
NUMERIC_COLS = [
    "dur", "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]

TCP_FEATURES = ["swin", "dwin", "tcprtt", "synack", "ackdat"]
CT_FEATURES = [
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd",
]

ATTACK_CATEGORIES = [
    "Fuzzers", "Analysis", "Backdoors", "DoS", "Exploits",
    "Generic", "Reconnaissance", "Shellcode", "Worms",
]

FOOTER = "Dataset: UNSW-NB15 | Source: UNSW Canberra Cyber | Created for: Network Intrusion Detection Research"

RAW_COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label",
]

TRAIN_TEST_COLS_MAP = {
    "sload": "Sload",
    "dload": "Dload",
    "spkts": "Spkts",
    "dpkts": "Dpkts",
    "sinpkt": "Sintpkt",
    "dinpkt": "Dintpkt",
    "sjit": "Sjit",
    "djit": "Djit",
    "smean": "smeansz",
    "dmean": "dmeansz",
    "response_body_len": "res_bdy_len",
    "label": "Label",
}
