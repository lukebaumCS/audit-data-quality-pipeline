import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.etl import (
    load_raw_data,
    standardize_types,
    flag_duplicates,
    flag_missing_required_fields,
    write_to_db,
    run_etl,
)


class TestETL(unittest.TestCase):

    def setUp(self):
        """Test data required for several tests."""
        self.df = pd.DataFrame(
            {
                "transaction_id": ["1", "2", "3", "4"],
                "account": ["1000", "1000", "2000", "3000"],
                "contra_account": ["2000", "2000", "4000", "5000"],
                "amount": ["100.50", "100.50", "200.00", "300.00"],
                "cost_center": ["CC01", "CC01", "CC02", "CC03"],
                "clerk": ["Max", "Max", "Anna", "Peter"],
                "currency": ["EUR", "EUR", "EUR", "EUR"],
                "payment_type": [
                    "Überweisung",
                    "Überweisung",
                    "Überweisung",
                    "Überweisung",
                ],
                "text": [
                    " Zahlung A ",
                    "Zahlung A",
                    " Zahlung B ",
                    "Zahlung C",
                ],
                "booking_datetime": [
                    "2025-01-15 10:00:00",
                    "2025-01-15 11:00:00",
                    "2025-01-16 12:00:00",
                    "2025-01-17 13:00:00",
                ],
                "document_number": ["DOC1", "DOC2", "DOC3", "DOC4"],
            }
        )

    # LOAD / EXTRACT

    def test_load_raw_data(self):
        """The CSV file is being read correctly."""
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "transactions.csv"

            self.df.to_csv(csv_path, index=False)

            result = load_raw_data(csv_path)

            self.assertEqual(len(result), 4)
            self.assertEqual(result.loc[0, "account"], "1000")

            # Certain columns should be read as strings
            self.assertEqual(
                result["account"].dtype.name,
                "string",
            )

            self.assertEqual(
                result["transaction_id"].dtype.name,
                "string",
            )

    def test_load_raw_data_file_not_found(self):
        """A non-existent CSV file triggers a FileNotFoundError."""
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "does_not_exist.csv"

            with self.assertRaises(FileNotFoundError):
                load_raw_data(csv_path)

    # TRANSFORM – TYPES


    def test_standardize_types(self):
        """Datentypen werden korrekt standardisiert."""
        result = standardize_types(self.df)

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                result["booking_datetime"]
            )
        )

        self.assertTrue(
            pd.api.types.is_numeric_dtype(
                result["amount"]
            )
        )

        self.assertEqual(
            result.loc[0, "booking_date"],
            pd.Timestamp("2025-01-15").date(),
        )

    def test_standardize_types_strips_whitespace(self):
        """Whitespace in string fields is removed."""
        result = standardize_types(self.df)

        self.assertEqual(
            result.loc[0, "text"],
            "Zahlung A",
        )

        self.assertEqual(
            result.loc[0, "currency"],
            "EUR",
        )

    def test_standardize_types_invalid_values(self):
        """Invalid date/numeric values are converted to NaN/NaT."""
        df = self.df.copy()

        df.loc[0, "booking_datetime"] = "invalid-date"
        df.loc[0, "amount"] = "invalid-amount"

        result = standardize_types(df)

        self.assertTrue(
            pd.isna(result.loc[0, "booking_datetime"])
        )

        self.assertTrue(
            pd.isna(result.loc[0, "booking_date"])
        )

        self.assertTrue(
            pd.isna(result.loc[0, "amount"])
        )

    # TRANSFORM – DUPLICATES

    def test_flag_duplicates(self):
        """Identical technical bookings are recognized as duplicates."""
        df = standardize_types(self.df)

        result = flag_duplicates(df)

        self.assertTrue(
            result.loc[0, "is_potential_duplicate"]
        )

        self.assertTrue(
            result.loc[1, "is_potential_duplicate"]
        )

        self.assertFalse(
            result.loc[2, "is_potential_duplicate"]
        )

        self.assertFalse(
            result.loc[3, "is_potential_duplicate"]
        )

    def test_duplicate_group_size(self):
        """The size of a duplicate group is calculated correctly."""
        df = standardize_types(self.df)

        result = flag_duplicates(df)

        self.assertEqual(
            result.loc[0, "duplicate_group_size"],
            2,
        )

        self.assertEqual(
            result.loc[1, "duplicate_group_size"],
            2,
        )

        self.assertEqual(
            result.loc[2, "duplicate_group_size"],
            1,
        )

        self.assertEqual(
            result.loc[3, "duplicate_group_size"],
            1,
        )

    def test_same_transaction_different_date_is_not_duplicate(self):
        """Identical transactions on different dates are not duplicates."""
        df = pd.DataFrame(
            {
                "account": ["1000", "1000"],
                "contra_account": ["2000", "2000"],
                "amount": [100.0, 100.0],
                "cost_center": ["CC01", "CC01"],
                "clerk": ["Max", "Max"],
                "booking_date": [
                    pd.Timestamp("2025-01-01").date(),
                    pd.Timestamp("2025-01-02").date(),
                ],
            }
        )

        result = flag_duplicates(df)

        self.assertFalse(
            result.loc[0, "is_potential_duplicate"]
        )

        self.assertFalse(
            result.loc[1, "is_potential_duplicate"]
        )

    # DATA QUALITY

    def test_flag_missing_required_fields(self):
        """Missing required fields are detected."""
        df = pd.DataFrame(
            {
                "account": ["1000", None, "3000"],
                "amount": [100.0, 200.0, None],
                "cost_center": ["CC01", "CC02", "CC03"],
                "clerk": ["Max", "Anna", None],
                "booking_datetime": [
                    pd.Timestamp("2025-01-01"),
                    pd.Timestamp("2025-01-02"),
                    pd.Timestamp("2025-01-03"),
                ],
            }
        )

        result = flag_missing_required_fields(df)

        self.assertFalse(
            result.loc[0, "has_missing_required_field"]
        )

        self.assertTrue(
            result.loc[1, "has_missing_required_field"]
        )

        self.assertTrue(
            result.loc[2, "has_missing_required_field"]
        )

    def test_missing_booking_datetime_is_detected(self):
        """Missing booking date is detected as an error."""
        df = pd.DataFrame(
            {
                "account": ["1000"],
                "amount": [100.0],
                "cost_center": ["CC01"],
                "clerk": ["Max"],
                "booking_datetime": [pd.NaT],
            }
        )

        result = flag_missing_required_fields(df)

        self.assertTrue(
            result.loc[0, "has_missing_required_field"]
        )

    # LOAD – DATABASE

    def test_write_to_db(self):
        """DataFrame is written correctly to SQLite."""
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "transactions.db"

            df = standardize_types(self.df)
            df = flag_duplicates(df)
            df = flag_missing_required_fields(df)

            write_to_db(
                df,
                db_path,
                "transactions_clean",
            )

            self.assertTrue(db_path.exists())

            with sqlite3.connect(db_path) as conn:
                result = pd.read_sql(
                    "SELECT * FROM transactions_clean",
                    conn,
                )

            self.assertEqual(len(result), 4)

            self.assertIn(
                "is_potential_duplicate",
                result.columns,
            )

            self.assertIn(
                "duplicate_group_size",
                result.columns,
            )

            self.assertIn(
                "has_missing_required_field",
                result.columns,
            )

    def test_write_to_db_replaces_existing_table(self):
        """Existing table is replaced when writing to SQLite."""
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "transactions.db"

            df = standardize_types(self.df)
            df = flag_duplicates(df)
            df = flag_missing_required_fields(df)

            # Start by writing just two lines
            write_to_db(
                df.iloc[:2],
                db_path,
                "transactions_clean",
            )

            # Then write four lines
            write_to_db(
                df,
                db_path,
                "transactions_clean",
            )

            with sqlite3.connect(db_path) as conn:
                result = pd.read_sql(
                    "SELECT * FROM transactions_clean",
                    conn,
                )

            self.assertEqual(len(result), 4)

    # END-TO-END

    def test_run_etl(self):
        """The complete ETL pipeline works."""
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "transactions.csv"
            db_path = Path(tmp_dir) / "transactions.db"

            self.df.to_csv(
                input_path,
                index=False,
            )

            result = run_etl(
                input_path,
                db_path,
                "transactions_clean",
            )

            self.assertIsInstance(
                result,
                pd.DataFrame,
            )

            self.assertEqual(
                len(result),
                4,
            )

            self.assertIn(
                "booking_date",
                result.columns,
            )

            self.assertIn(
                "is_potential_duplicate",
                result.columns,
            )

            self.assertIn(
                "duplicate_group_size",
                result.columns,
            )

            self.assertIn(
                "has_missing_required_field",
                result.columns,
            )

            self.assertTrue(
                db_path.exists()
            )

    def test_run_etl_detects_missing_fields(self):
        """End-to-End: missing required field is detected."""
        with TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "transactions.csv"
            db_path = Path(tmp_dir) / "transactions.db"

            df = self.df.copy()
            df.loc[2, "account"] = None

            df.to_csv(
                input_path,
                index=False,
            )

            result = run_etl(
                input_path,
                db_path,
            )

            self.assertTrue(
                result.loc[2, "has_missing_required_field"]
            )


if __name__ == "__main__":
    unittest.main()