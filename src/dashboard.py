from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Audit Anomaly Detection Dashboard", layout="wide")


# Data sources (sidebar, configurable)

st.sidebar.header("Datenquelle")
db_path = st.sidebar.text_input("SQLite-Datenbank", "data/processed/transactions.db")
table_name = st.sidebar.text_input("Tabelle", "transactions_flagged")
dq_log_path = st.sidebar.text_input("Data-Quality-Log", "data/processed/dq_score_history.csv")


@st.cache_data
def load_transactions(db_path: str, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"], errors="coerce")
    return df


@st.cache_data
def load_dq_log(path: str) -> pd.DataFrame:
    log_path = Path(path)
    if not log_path.exists():
        return pd.DataFrame()
    return pd.read_csv(log_path)


if not Path(db_path).exists():
    st.error(
        f"Datenbank nicht gefunden: `{db_path}`.\n\n"
        "Erst die Pipeline laufen lassen: generator.py -> etl.py -> "
        "data_quality.py -> anomaly_detection.py."
    )
    st.stop()

df = load_transactions(db_path, table_name)
dq_log = load_dq_log(dq_log_path)


# Key figures

st.title("Audit Data Quality & Anomaly Detection Dashboard")

total_transactions = len(df)
flagged_count = int(df["is_flagged"].sum()) if "is_flagged" in df.columns else 0
anomaly_rate = (flagged_count / total_transactions * 100) if total_transactions else 0.0
latest_dq_score = dq_log["overall_score_pct"].iloc[-1] if not dq_log.empty else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transaktionen", f"{total_transactions:,}")
col2.metric("Erkannte Anomalien", f"{flagged_count:,}")
col3.metric("Anomalierate", f"{anomaly_rate:.2f}%")
col4.metric(
    "Data-Quality-Score",
    f"{latest_dq_score:.2f}%" if latest_dq_score is not None else "n/a",
)

st.divider()


# Anomalies by category

st.subheader("Anomalien nach Kategorie")
rule_columns = [c for c in df.columns if c.startswith("rule_")]

if rule_columns:
    rule_counts = df[rule_columns].sum().sort_values(ascending=False)
    rule_counts.index = [name.replace("rule_", "") for name in rule_counts.index]
    fig_rules = px.bar(
        rule_counts,
        labels={"index": "Regel", "value": "Anzahl Treffer"},
        text_auto=True,
    )
    fig_rules.update_layout(showlegend=False, yaxis_title="Anzahl Treffer", xaxis_title="")
    st.plotly_chart(fig_rules, use_container_width=True)
else:
    st.info("Keine Regel-Spalten gefunden — erst anomaly_detection.py ausführen.")


# Trends over time

st.subheader("Zeitliche Entwicklung")

if "booking_datetime" in df.columns and "is_flagged" in df.columns:
    monthly = df.copy()
    monthly["booking_month"] = monthly["booking_datetime"].dt.to_period("M").astype(str)
    monthly_summary = (
        monthly.groupby("booking_month")
        .agg(Transaktionen=("booking_month", "size"), Anomalien=("is_flagged", "sum"))
        .reset_index()
        .sort_values("booking_month")
    )
    fig_time = px.line(
        monthly_summary,
        x="booking_month",
        y=["Transaktionen", "Anomalien"],
        markers=True,
        labels={"booking_month": "Monat", "value": "Anzahl", "variable": ""},
    )
    st.plotly_chart(fig_time, use_container_width=True)


# High-risk transactions

st.subheader("Top-Risikotransaktionen")

if "rules_triggered_count" in df.columns:
    display_columns = [
        "transaction_id", "booking_datetime", "account", "contra_account",
        "amount", "clerk", "triggered_rules", "rules_triggered_count",
    ]
    display_columns = [c for c in display_columns if c in df.columns]

    top_risk = df.sort_values("rules_triggered_count", ascending=False).head(20)
    st.dataframe(top_risk[display_columns], use_container_width=True, hide_index=True)
    st.caption(
        "Aktuell sortiert nach Anzahl ausgelöster Regeln. "
        "Ein gewichteter Risk Score mit Begründung folgt in Phase 2.3."
    )


# Data Quality Score History

st.subheader("Data-Quality-Score über mehrere Läufe")

if not dq_log.empty:
    fig_dq = px.line(
        dq_log,
        x="run_timestamp",
        y="overall_score_pct",
        markers=True,
        labels={"run_timestamp": "Lauf", "overall_score_pct": "Score (%)"},
    )
    fig_dq.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_dq, use_container_width=True)
else:
    st.info("Noch keine Data-Quality-Läufe geloggt — erst data_quality.py ausführen.")
