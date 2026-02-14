import io
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Defaults

@dataclass
class Config:
    amount_z_threshold: float = 3.5 # z-score threshold for amount outlier
    sum24h_z_threshold: float = 3.5 # z-score threshold for 24h sum spike
    velocity_count_1h_threshold: int = 8 # Burst flag min tx count in 1h

    score_amount_outlier: int = 40 # Amount outlier score
    score_sum24h_spike: int = 30 # 24h sum spike score
    score_velocity_burst: int = 30 # Velocity burst score
    score_new_counterparty: int = 20 # New counterparty score
    score_duplicate_txid: int = 10 # Duplicate transaction_id score

    score_cap: int = 100 # Max total score per transaction

REQUIRED_COLS = ["timestamp", "account_id", "amount"]
OPTIONAL_COLS = ["transaction_id", "counterparty", "currency", "direction", "channel", "country"]

# Utility

def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Robust z-score using Median Absolute Deviation (MAD).
    z = 0.6745 * (x - median) / MAD
    """
    x = series.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    mad = mad if mad and mad > 0 else 1e-9
    return 0.6745 * (x - med) / mad

def per_group_robust_z(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    return df.groupby(group_col)[value_col].transform(robust_zscore)

def parse_transactions_csv(file_bytes: bytes) -> Tuple[pd.DataFrame, List[str]]:
    """
    Parses CSV bytes into a DataFrame
    attempts to coerce timestamp
    returns (df, warnings)
    """
    warnings = []
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Check required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {REQUIRED_COLS}")

    # Parse timestamps
    df["timestamp_raw"] = df["timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    bad_ts = df["timestamp"].isna().sum()
    if bad_ts > 0:
        warnings.append(f"{bad_ts} rows have invalid timestamps and will be dropped.")

    # Coerce amount
    df["amount_raw"] = df["amount"]
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    bad_amt = df["amount"].isna().sum()
    if bad_amt > 0:
        warnings.append(f"{bad_amt} rows have invalid amounts and will be dropped.")

    # Drop invalid requireds
    df = df.dropna(subset=["timestamp", "account_id", "amount"]).copy()

    # Check account_id is a string
    df["account_id"] = df["account_id"].astype(str)

    # Add missing optionals as None
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None

    return df, warnings

def generate_synthetic_transactions(
    n_accounts: int = 20,
    days: int = 7,
    seed: int = 42,
    inject_anomalies: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    accounts = [f"ACC_{i:03d}" for i in range(n_accounts)]
    counterparties = [f"CP_{i:03d}" for i in range(60)]

    rows = []
    base_start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(days=days)

    tx_id = 0
    for acc in accounts:
        # Rand activity rate per account
        daily_tx = rng.integers(5, 25)
        n_tx = int(daily_tx * days)

        # Generate timestamps
        times = base_start + pd.to_timedelta(rng.uniform(0, days * 24 * 60, size=n_tx), unit="m")
        times = pd.to_datetime(times).sort_values()

        # Generate amounts
        # Each account has own scale
        scale = float(rng.uniform(10, 500))
        amounts = rng.lognormal(mean=math.log(max(scale, 1.0)), sigma=0.8, size=n_tx)
        # Add signs based on direction
        direction = rng.choice(["out", "in"], size=n_tx, p=[0.7, 0.3])
        signed_amounts = np.where(direction == "out", -amounts, amounts)

        cps = rng.choice(counterparties, size=n_tx, replace=True)

        for t, a, d, cp in zip(times, signed_amounts, direction, cps):
            tx_id += 1
            rows.append(
                {
                    "transaction_id": f"TX_{tx_id:08d}",
                    "timestamp": t.isoformat(),
                    "account_id": acc,
                    "counterparty": cp,
                    "amount": float(a),
                    "currency": "GBP",
                    "direction": d,
                    "channel": rng.choice(["card", "cash"], p=[0.4, 0.5, 0.1]),
                    "country": rng.choice(["GB", "US", "DE", "PT", "FR"], p=[0.5, 0.15, 0.1, 0.15, 0.1]),
                }
            )

    df = pd.DataFrame(rows)

    if inject_anomalies and len(df) > 0:
        # Huge amount outlier
        i = rng.integers(0, len(df))
        df.loc[i, "amount"] = float(df["amount"].abs().median() * 80) * (-1 if rng.random() < 0.7 else 1)

        # Burst of many small tx in a short period for one account
        acc = df["account_id"].iloc[rng.integers(0, len(df))]
        burst_start = pd.Timestamp.utcnow().floor("min") - pd.Timedelta(hours=2)
        burst_rows = []
        for _ in range(15):
            tx_id += 1
            burst_rows.append(
                {
                    "transaction_id": f"TX_{tx_id:08d}",
                    "timestamp": (burst_start + pd.Timedelta(minutes=int(rng.uniform(0, 55)))).isoformat(),
                    "account_id": acc,
                    "counterparty": rng.choice(counterparties),
                    "amount": float(rng.uniform(-40, -5)),  # small outflows
                    "currency": "GBP",
                    "direction": "out",
                    "channel": "card",
                    "country": "GB",
                }
            )
        df = pd.concat([df, pd.DataFrame(burst_rows)], ignore_index=True)

        # New counterparty + new country combo
        i2 = rng.integers(0, len(df))
        df.loc[i2, "counterparty"] = "CP_NEW_999"
        df.loc[i2, "country"] = "ZZ"

        # Duplicate transaction_id
        if len(df) >= 2:
            df.loc[len(df) - 1, "transaction_id"] = df.loc[len(df) - 2, "transaction_id"]

    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - amount_abs
      - tx_count_1h per account
      - amt_sum_24h per account
    Uses groupby apply with time-based rolling.
    """
    df = df.copy()
    df["row_id"] = np.arange(len(df))
    df = df.sort_values(["account_id", "timestamp"])

    df["amount_abs"] = df["amount"].abs()
    # log1p for stability
    df["log_amount_abs"] = np.log1p(df["amount_abs"].clip(lower=0.0))

    def _roll(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        g = g.set_index("timestamp")
        g["tx_count_1h"] = g["amount_abs"].rolling("1H").count()
        g["amt_sum_24h"] = g["amount_abs"].rolling("24H").sum()
        return g.reset_index()

    out = df.groupby("account_id", group_keys=False).apply(_roll)
    out = out.sort_values("row_id").drop(columns=["row_id"])
    return out

def score_anomalies(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Adds columns:
      - amount_z, sum24h_z
      - is_new_counterparty, is_duplicate_txid
      - score, severity, reasons
      - is_flagged
    """
    df = df.copy()

    # Amount outlier per account
    df["amount_z"] = per_group_robust_z(df, "account_id", "log_amount_abs").abs()

    # 24h sum spike per account
    df["log_sum24h"] = np.log1p(df["amt_sum_24h"].clip(lower=0.0))
    df["sum24h_z"] = per_group_robust_z(df, "account_id", "log_sum24h").abs()

    # New counterparty per account (if provided)
    if df["counterparty"].notna().any():
        # Treat None/NaN as not new
        df["counterparty_filled"] = df["counterparty"].fillna("")
        df["is_new_counterparty"] = df.groupby("account_id")["counterparty_filled"].transform(lambda s: ~s.duplicated())
        df.loc[df["counterparty_filled"] == "", "is_new_counterparty"] = False
    else:
        df["is_new_counterparty"] = False

    # Duplicate transaction_id (if provided)
    if df["transaction_id"].notna().any():
        df["is_duplicate_txid"] = df["transaction_id"].notna() & df.duplicated("transaction_id", keep=False)
    else:
        df["is_duplicate_txid"] = False

    # Treat missing as 0
    df["is_velocity_burst"] = df["tx_count_1h"].fillna(0).astype(float) >= float(cfg.velocity_count_1h_threshold)

    # Apply rules + score + reasons
    scores = []
    severities = []
    reasons_out: List[List[str]] = []
    flagged = []

    for _, r in df.iterrows():
        score = 0
        reasons = []

        if bool(r.get("is_duplicate_txid", False)):
            score += cfg.score_duplicate_txid
            reasons.append("Duplicate transaction_id")

        if float(r.get("amount_z", 0.0)) >= cfg.amount_z_threshold:
            score += cfg.score_amount_outlier
            reasons.append(f"Unusual amount for this account (robust_z={r['amount_z']:.2f})")

        if bool(r.get("is_velocity_burst", False)):
            score += cfg.score_velocity_burst
            reasons.append(f"High transaction velocity (tx_count_1h={int(r['tx_count_1h'])})")

        if float(r.get("sum24h_z", 0.0)) >= cfg.sum24h_z_threshold:
            score += cfg.score_sum24h_spike
            reasons.append(f"Unusually high 24h total (robust_z={r['sum24h_z']:.2f})")

        if bool(r.get("is_new_counterparty", False)):
            score += cfg.score_new_counterparty
            reasons.append("New counterparty for this account")

        score = min(score, cfg.score_cap)

        if score >= 80:
            sev = "High"
        elif score >= 50:
            sev = "Medium"
        elif score > 0:
            sev = "Low"
        else:
            sev = "None"

        is_flagged = score > 0

        scores.append(score)
        severities.append(sev)
        reasons_out.append(reasons)
        flagged.append(is_flagged)

    df["score"] = scores
    df["severity"] = severities
    df["reasons"] = reasons_out
    df["is_flagged"] = flagged

    # Clean helper col
    if "counterparty_filled" in df.columns:
        df = df.drop(columns=["counterparty_filled"])

    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# Streamlit UI

st.set_page_config(page_title="Transaction Anomaly Spotter", layout="wide")
st.title("Transaction Anomaly Spotter")
st.caption("Upload a transactions CSV (or generate a sample dataset) and get flagged anomalies.")

with st.sidebar:
    st.header("Detection Settings")
    cfg = Config(
        amount_z_threshold=st.slider("Amount outlier threshold (robust z)", 2.0, 6.0, 3.5, 0.1),
        sum24h_z_threshold=st.slider("24h sum spike threshold (robust z)", 2.0, 6.0, 3.5, 0.1),
        velocity_count_1h_threshold=st.slider("Velocity threshold (tx count in last 1h)", 2, 30, 8, 1),
    )
    st.divider()
    st.write("Expected minimum columns:")
    st.code(", ".join(REQUIRED_COLS), language="text")
    st.write("Optional columns (improves signals):")
    st.code(", ".join(OPTIONAL_COLS), language="text")

colA, colB = st.columns([2, 1])

with colB:
    st.subheader("Demo data")
    demo_seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)
    demo_days = st.slider("Days", 1, 30, 7, 1)
    demo_accounts = st.slider("Accounts", 5, 200, 20, 5)
    inject = st.checkbox("Inject anomalies", value=True)

    if st.button("Generate synthetic CSV", use_container_width=True):
        demo_df = generate_synthetic_transactions(
            n_accounts=int(demo_accounts), days=int(demo_days), seed=int(demo_seed), inject_anomalies=inject
        )
        st.download_button(
            "Download demo_transactions.csv",
            data=df_to_csv_bytes(demo_df),
            file_name="demo_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )

with colA:
    st.subheader("Upload CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to analyse, or generate demo data from the right panel.")
        st.stop()

    raw_bytes = uploaded.getvalue()

    try:
        df, warnings = parse_transactions_csv(raw_bytes)
    except Exception as e:
        st.error(str(e))
        st.stop()

    for w in warnings:
        st.warning(w)

    if df.empty:
        st.warning("No valid rows after parsing. Check your CSV format.")
        st.stop()

    # Add features + score
    df = df.sort_values("timestamp")
    df_feat = add_rolling_features(df)
    df_scored = score_anomalies(df_feat, cfg)

    # Summary metrics
    flagged_df = df_scored[df_scored["is_flagged"]].copy()
    high = (flagged_df["severity"] == "High").sum()
    med = (flagged_df["severity"] == "Medium").sum()
    low = (flagged_df["severity"] == "Low").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", f"{len(df_scored):,}")
    m2.metric("Flagged", f"{len(flagged_df):,}")
    m3.metric("High / Medium / Low", f"{high} / {med} / {low}")
    m4.metric("Accounts", f"{df_scored['account_id'].nunique():,}")

    st.divider()

    # Output table
    st.subheader("Flagged transactions")
    show_cols = [
        "timestamp",
        "transaction_id",
        "account_id",
        "counterparty",
        "amount",
        "currency",
        "direction",
        "tx_count_1h",
        "amt_sum_24h",
        "amount_z",
        "sum24h_z",
        "score",
        "severity",
        "reasons",
    ]
    show_cols = [c for c in show_cols if c in flagged_df.columns]

    st.dataframe(
        flagged_df.sort_values(["severity", "score"], ascending=[False, False])[show_cols],
        use_container_width=True,
        height=420,
    )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download flagged_results.csv",
            data=df_to_csv_bytes(flagged_df[show_cols]),
            file_name="flagged_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download full_scored.csv",
            data=df_to_csv_bytes(df_scored),
            file_name="full_scored.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.divider()

    # Charts
    st.subheader("Quick visuals")

    c1, c2 = st.columns(2)

    with c1:
        st.write("Amount distribution (absolute)")
        fig = plt.figure()
        vals = df_scored["amount_abs"].replace([np.inf, -np.inf], np.nan).dropna()
        plt.hist(vals, bins=40)
        plt.xlabel("Amount (abs)")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.write("Flagged count over time (hourly)")
        tmp = df_scored.copy()
        tmp["is_flagged_int"] = tmp["is_flagged"].astype(int)
        tmp = tmp.set_index("timestamp").sort_index()
        hourly = tmp["is_flagged_int"].resample("1H").sum()
        fig2 = plt.figure()
        plt.plot(hourly.index, hourly.values)
        plt.xlabel("Time")
        plt.ylabel("Flagged count")
        st.pyplot(fig2, clear_figure=True)
