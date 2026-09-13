from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# Fields that constitute the ‘technical identity’ of a transaction.
# Two rows with identical values in these fields are considered
# potential duplicates (regardless of the generated ground-truth label).
DUPLICATE_KEY_FIELDS = ["account", "contra_account", "amount", "cost_center", "clerk"]


# Extract

def load_raw_data(source: str | Path) -> pd.DataFrame:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Rohdatendatei nicht gefunden: {source}")

    dtype_overrides = {
        "account": "string",
        "contra_account": "string",
        "cost_center": "string",
        "document_number": "string",
        "transaction_id": "string",
    }
    return pd.read_csv(source, dtype=dtype_overrides)


# Transform — Standardise data types

def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"], errors="coerce")
    df["booking_date"] = df["booking_datetime"].dt.date

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Accounts as fixed-width strings, preserving leading zeros    
    for col in ["account", "contra_account"]:
        df[col] = df[col].astype("string").str.strip()

    for col in ["currency", "cost_center", "clerk", "payment_type"]:
        df[col] = df[col].astype("string").str.strip()

    df["text"] = df["text"].astype("string").str.strip()

    return df


# Transform — Detect duplicates
def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Markiert fachliche Duplikate: gleiche Kernfelder + gleiches Buchungsdatum.

    Bewusst ohne exakte Uhrzeit im Schlüssel, da echte Doppelbuchungen oft
    minimal versetzt eingegeben werden.
    """
    df = df.copy()
    key_fields = DUPLICATE_KEY_FIELDS + ["booking_date"]

    duplicate_mask = df.duplicated(subset=key_fields, keep=False)
    df["is_potential_duplicate"] = duplicate_mask

    # Group size as additional context (e.g. 3 rather than just 2 identical bookings)
    group_sizes = df.groupby(key_fields)[key_fields[0]].transform("size")
    df["duplicate_group_size"] = group_sizes.where(duplicate_mask, 1)

    return df


# Transform — simple data quality flags

def flag_missing_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["account", "amount", "cost_center", "clerk", "booking_datetime"]
    df["has_missing_required_field"] = df[required].isna().any(axis=1)
    return df



# Load — write to an SQL database

def write_to_db(
    df: pd.DataFrame,
    db_path: str | Path,
    table_name: str = "transactions_clean",
) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Date objects for SQLite as ISO strings; pandas retains timestamps
    df_to_write = df.copy()
    df_to_write["booking_date"] = df_to_write["booking_date"].astype(str)

    with sqlite3.connect(db_path) as conn:
        df_to_write.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"{len(df):,} Zeilen geschrieben -> {db_path} (Tabelle '{table_name}')")


# Orchestration

def run_etl(input_path: str | Path, db_path: str | Path, table_name: str = "transactions_clean") -> pd.DataFrame:
    df = load_raw_data(input_path)
    df = standardize_types(df)
    df = flag_duplicates(df)
    df = flag_missing_required_fields(df)
    write_to_db(df, db_path, table_name)

    print(f"Potenzielle Duplikate erkannt: {df['is_potential_duplicate'].sum():,}")
    print(f"Zeilen mit fehlenden Pflichtfeldern: {df['has_missing_required_field'].sum():,}")

    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Einfache ETL-Pipeline für Buchungssätze")
    parser.add_argument("--input", type=str, default="data/raw/transactions.csv")
    parser.add_argument("--db", type=str, default="data/processed/transactions.db")
    parser.add_argument("--table-name", type=str, default="transactions_clean")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_etl(args.input, args.db, args.table_name)


if __name__ == "__main__":
    main()
