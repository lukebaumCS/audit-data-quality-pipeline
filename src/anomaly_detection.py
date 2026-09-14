from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AnomalyDetectionConfig:
    business_hour_start: int = 6
    business_hour_end: int = 20
    round_amount_threshold: float = 500.0       # At this level, ‘around’ is considered conspicuous
    round_amount_divisor: float = 500.0         # must be divisible by this value without a remainder
    high_amount_quantile: float = 0.99          # top 1 % is considered unusually high
    unusual_combo_min_frequency: float = 0.02   # account must appear in at least 2 % of this clerk's transactions


RULE_NAMES = [
    "rule_duplicate",
    "rule_weekend_night",
    "rule_round_amount",
    "rule_high_amount",
    "rule_unusual_account_combo",
]


class RuleBasedAnomalyDetector:
    def __init__(self, config: AnomalyDetectionConfig | None = None) -> None:
        self.config = config or AnomalyDetectionConfig()


    def detect_duplicates(self, df: pd.DataFrame) -> pd.Series:
        """Übernimmt das Ergebnis aus der ETL-Duplikaterkennung, falls vorhanden."""
        if "is_potential_duplicate" in df.columns:
            return df["is_potential_duplicate"].astype(bool)
        key_fields = ["account", "contra_account", "amount", "cost_center", "clerk", "booking_date"]
        return df.duplicated(subset=key_fields, keep=False)

    def detect_weekend_night(self, df: pd.DataFrame) -> pd.Series:
        dt = pd.to_datetime(df["booking_datetime"], errors="coerce")
        is_weekend = dt.dt.dayofweek >= 5
        is_night = ~dt.dt.hour.between(self.config.business_hour_start, self.config.business_hour_end)
        return dt.notna() & (is_weekend | is_night)

    def detect_round_amount(self, df: pd.DataFrame) -> pd.Series:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        is_large_enough = amount >= self.config.round_amount_threshold
        is_round = (amount % self.config.round_amount_divisor) == 0
        return amount.notna() & is_large_enough & is_round

    def detect_high_amount(self, df: pd.DataFrame) -> pd.Series:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        threshold = amount.quantile(self.config.high_amount_quantile)
        return amount.notna() & (amount > threshold)

    def detect_unusual_account_combo(self, df: pd.DataFrame) -> pd.Series:
        """Data-driven: how often does this clerk normally make entries to this account?"""
        clerk_account_count = df.groupby(["clerk", "account"])["clerk"].transform("size")
        clerk_total_count = df.groupby("clerk")["clerk"].transform("size")
        relative_frequency = clerk_account_count / clerk_total_count
        return relative_frequency < self.config.unusual_combo_min_frequency


    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["rule_duplicate"] = self.detect_duplicates(df)
        df["rule_weekend_night"] = self.detect_weekend_night(df)
        df["rule_round_amount"] = self.detect_round_amount(df)
        df["rule_high_amount"] = self.detect_high_amount(df)
        df["rule_unusual_account_combo"] = self.detect_unusual_account_combo(df)

        df["rules_triggered_count"] = df[RULE_NAMES].sum(axis=1)
        df["is_flagged"] = df["rules_triggered_count"] > 0
        df["triggered_rules"] = df[RULE_NAMES].apply(
            lambda row: ",".join(name.replace("rule_", "") for name, hit in row.items() if hit),
            axis=1,
        )
        return df

    def summarize(self, df: pd.DataFrame) -> dict:
        summary = {
            "total_rows": len(df),
            "flagged_rows": int(df["is_flagged"].sum()),
            "flagged_rate_pct": round(df["is_flagged"].mean() * 100, 2),
        }
        for rule_name in RULE_NAMES:
            summary[f"{rule_name}_count"] = int(df[rule_name].sum())
        return summary


def quick_ground_truth_check(df: pd.DataFrame) -> None:
    if "is_anomaly" not in df.columns:
        return

    true_anomalies = df["is_anomaly"].astype(bool)
    caught = df["is_flagged"] & true_anomalies
    missed = true_anomalies & ~df["is_flagged"]
    false_alarms = df["is_flagged"] & ~true_anomalies

    print("\n--- Informelle Gegenprobe (kein Ersatz für Phase 2.4 Evaluation) ---")
    print(f"Bekannte Anomalien im Datensatz: {true_anomalies.sum():,}")
    print(f"Davon von Regeln erfasst:        {caught.sum():,}")
    print(f"Davon verpasst:                  {missed.sum():,}")
    print(f"Fehlalarme (geflaggt, aber nicht echt): {false_alarms.sum():,}")


# I/O

def load_from_db(db_path: str | Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def write_to_db(df: pd.DataFrame, db_path: str | Path, table_name: str) -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


# CLI

def run_anomaly_detection(
    db_path: str | Path,
    table_name: str = "transactions_clean",
    output_table: str = "transactions_flagged",
) -> pd.DataFrame:
    df = load_from_db(db_path, table_name)
    detector = RuleBasedAnomalyDetector()
    df = detector.run(df)
    write_to_db(df, db_path, output_table)

    summary = detector.summarize(df)
    print(f"Geflaggte Transaktionen: {summary['flagged_rows']:,} ({summary['flagged_rate_pct']}%)")
    for rule_name in RULE_NAMES:
        print(f"  {rule_name}: {summary[f'{rule_name}_count']:,} Treffer")

    quick_ground_truth_check(df)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regelbasierte Anomalie-Erkennung")
    parser.add_argument("--db", type=str, default="data/processed/transactions.db")
    parser.add_argument("--table", type=str, default="transactions_clean")
    parser.add_argument("--output-table", type=str, default="transactions_flagged")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_anomaly_detection(args.db, args.table, args.output_table)


if __name__ == "__main__":
    main()
