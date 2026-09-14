import sqlite3
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_quality import (
    CHECK_NAMES,
    DataQualityConfig,
    DataQualityChecker,
    append_summary_log,
    load_from_db,
    run_data_quality,
)


class TestDataQualityChecker(unittest.TestCase):

    def setUp(self):
        """Create a valid baseline configuration and test DataFrame."""

        self.config = DataQualityConfig(
            min_amount=0.01,
            max_amount=500_000.0,
            valid_accounts=("1000", "2000", "3000"),
            valid_contra_accounts=("4000", "5000", "6000"),
            valid_currencies=("EUR", "USD"),
            valid_cost_centers=("CC01", "CC02"),
            valid_payment_types=("TRANSFER", "CARD"),
            date_min=datetime(2025, 1, 1),
            date_max=datetime(2025, 12, 31),
        )

        self.checker = DataQualityChecker(self.config)

        self.df = pd.DataFrame(
            {
                "account": ["1000", "2000", "3000"],
                "contra_account": ["4000", "5000", "6000"],
                "amount": [100.0, 200.0, 300.0],
                "cost_center": ["CC01", "CC02", "CC01"],
                "clerk": ["Max", "Anna", "Peter"],
                "currency": ["EUR", "EUR", "USD"],
                "payment_type": ["TRANSFER", "CARD", "TRANSFER"],
                "booking_datetime": [
                    "2025-01-15 10:00:00",
                    "2025-06-01 12:00:00",
                    "2025-12-31 23:59:59",
                ],
            }
        )

    # CONFIGURATION

    def test_config_uses_custom_values(self):
        """Custom configuration values should be stored correctly."""

        self.assertEqual(self.config.min_amount, 0.01)
        self.assertEqual(self.config.max_amount, 500_000.0)

        self.assertEqual(
            self.config.valid_accounts,
            ("1000", "2000", "3000"),
        )

        self.assertEqual(
            self.config.valid_currencies,
            ("EUR", "USD"),
        )

        self.assertEqual(
            self.config.date_min,
            datetime(2025, 1, 1),
        )

        self.assertEqual(
            self.config.date_max,
            datetime(2025, 12, 31),
        )

    # COMPLETENESS

    def test_check_completeness_with_etl_flag(self):
        """The completeness check should use the ETL-generated flag."""

        df = self.df.copy()

        df["has_missing_required_field"] = [
            False,
            True,
            False,
        ]

        result = self.checker.check_completeness(df)

        expected = pd.Series(
            [True, False, True],
            index=df.index,
        )

        pd.testing.assert_series_equal(
            result,
            expected,
            check_names=False,
        )

    def test_check_completeness_without_etl_flag(self):
        """The completeness check should inspect required fields directly."""

        df = self.df.copy()

        df.loc[1, "account"] = None
        df.loc[2, "amount"] = None

        result = self.checker.check_completeness(df)

        expected = pd.Series(
            [True, False, False],
            index=df.index,
        )

        pd.testing.assert_series_equal(
            result,
            expected,
            check_names=False,
        )

    # DUPLICATES

    def test_check_not_duplicate_with_etl_flag(self):
        """The duplicate check should use the ETL-generated flag."""

        df = self.df.copy()

        df["is_potential_duplicate"] = [
            False,
            True,
            False,
        ]

        result = self.checker.check_not_duplicate(df)

        expected = pd.Series(
            [True, False, True],
            index=df.index,
        )

        pd.testing.assert_series_equal(
            result,
            expected,
            check_names=False,
        )

    def test_check_not_duplicate_without_etl_flag(self):
        """Rows should pass when no duplicate flag is available."""

        result = self.checker.check_not_duplicate(self.df)

        self.assertTrue(result.all())
        self.assertEqual(len(result), len(self.df))

    # DATE VALIDATION

    def test_check_valid_date(self):
        """Dates inside the configured range should pass."""

        df = self.df.copy()

        df.loc[0, "booking_datetime"] = "2025-01-01"
        df.loc[1, "booking_datetime"] = "2025-06-15"
        df.loc[2, "booking_datetime"] = "2025-12-31"

        result = self.checker.check_valid_date(df)

        self.assertTrue(result.all())

    def test_check_invalid_date_outside_range(self):
        """Dates outside the configured range should fail."""

        df = self.df.copy()

        df.loc[0, "booking_datetime"] = "2024-12-31"
        df.loc[1, "booking_datetime"] = "2026-01-01"

        result = self.checker.check_valid_date(df)

        self.assertFalse(result.loc[0])
        self.assertFalse(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_invalid_date_format(self):
        """Invalid date values should fail validation."""

        df = self.df.copy()

        df.loc[0, "booking_datetime"] = "not-a-date"

        result = self.checker.check_valid_date(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    # AMOUNT VALIDATION

    def test_check_valid_amount(self):
        """Amounts inside the configured range should pass."""

        df = self.df.copy()

        df["amount"] = [
            0.01,
            100.00,
            500_000.00,
        ]

        result = self.checker.check_valid_amount(df)

        self.assertTrue(result.all())

    def test_check_invalid_amount_too_low(self):
        """Amounts below the minimum should fail."""

        df = self.df.copy()

        df.loc[0, "amount"] = 0.0

        result = self.checker.check_valid_amount(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_invalid_amount_too_high(self):
        """Amounts above the maximum should fail."""

        df = self.df.copy()

        df.loc[0, "amount"] = 500_000.01

        result = self.checker.check_valid_amount(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_invalid_amount_not_numeric(self):
        """Non-numeric amounts should fail validation."""

        df = self.df.copy()

        df.loc[0, "amount"] = "invalid"

        result = self.checker.check_valid_amount(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    # ACCOUNT VALIDATION

    def test_check_valid_accounts(self):
        """Valid account and contra-account combinations should pass."""

        result = self.checker.check_valid_accounts(self.df)

        self.assertTrue(result.all())

    def test_check_invalid_account(self):
        """An invalid account should cause the check to fail."""

        df = self.df.copy()

        df.loc[0, "account"] = "9999"

        result = self.checker.check_valid_accounts(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_invalid_contra_account(self):
        """An invalid contra-account should cause the check to fail."""

        df = self.df.copy()

        df.loc[0, "contra_account"] = "9999"

        result = self.checker.check_valid_accounts(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    # FIELD CONSISTENCY

    def test_check_field_consistency(self):
        """Valid field combinations should pass."""

        result = self.checker.check_field_consistency(self.df)

        self.assertTrue(result.all())

    def test_check_field_consistency_same_account(self):
        """Account and contra-account must not be identical."""

        df = self.df.copy()

        df.loc[0, "contra_account"] = df.loc[0, "account"]

        result = self.checker.check_field_consistency(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_field_consistency_invalid_currency(self):
        """An invalid currency should cause the check to fail."""

        df = self.df.copy()

        df.loc[0, "currency"] = "GBP"

        result = self.checker.check_field_consistency(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_field_consistency_invalid_cost_center(self):
        """An invalid cost center should cause the check to fail."""

        df = self.df.copy()

        df.loc[0, "cost_center"] = "INVALID"

        result = self.checker.check_field_consistency(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    def test_check_field_consistency_invalid_payment_type(self):
        """An invalid payment type should cause the check to fail."""

        df = self.df.copy()

        df.loc[0, "payment_type"] = "CASH"

        result = self.checker.check_field_consistency(df)

        self.assertFalse(result.loc[0])
        self.assertTrue(result.loc[1])
        self.assertTrue(result.loc[2])

    # RUN CHECKS

    def test_run_checks_creates_all_check_columns(self):
        """run_checks should create one result column per check."""

        result = self.checker.run_checks(self.df)

        for check_name in CHECK_NAMES:
            self.assertIn(check_name, result.columns)

        self.assertIn("dq_checks_passed", result.columns)
        self.assertIn("dq_row_score", result.columns)
        self.assertIn("dq_passes_all_checks", result.columns)

    def test_run_checks_valid_data_passes_all_checks(self):
        """A fully valid DataFrame should pass all checks."""

        result = self.checker.run_checks(self.df)

        self.assertTrue(
            result["dq_passes_all_checks"].all()
        )

        self.assertEqual(
            result["dq_checks_passed"].tolist(),
            [6, 6, 6],
        )

        self.assertEqual(
            result["dq_row_score"].tolist(),
            [1.0, 1.0, 1.0],
        )

    def test_run_checks_calculates_row_score(self):
        """The row score should equal passed checks divided by total checks."""

        df = self.df.copy()

        # Make the first row fail exactly one check.
        df.loc[0, "amount"] = 0

        result = self.checker.run_checks(df)

        self.assertEqual(
            result.loc[0, "dq_checks_passed"],
            5,
        )

        self.assertAlmostEqual(
            result.loc[0, "dq_row_score"],
            5 / 6,
        )

        self.assertFalse(
            result.loc[0, "dq_passes_all_checks"]
        )

    def test_run_checks_does_not_modify_original_dataframe(self):
        """run_checks should operate on a copy of the input DataFrame."""

        original_columns = self.df.columns.tolist()

        self.checker.run_checks(self.df)

        self.assertEqual(
            self.df.columns.tolist(),
            original_columns,
        )

    # SUMMARY

    def test_summarize(self):
        """The summary should contain overall and per-check pass rates."""

        df = self.checker.run_checks(self.df)

        summary = self.checker.summarize(df)

        self.assertEqual(
            summary["total_rows"],
            3,
        )

        self.assertEqual(
            summary["overall_score_pct"],
            100.0,
        )

        self.assertEqual(
            summary["rows_passing_all_checks_pct"],
            100.0,
        )

        for check_name in CHECK_NAMES:
            self.assertEqual(
                summary[f"{check_name}_pass_rate_pct"],
                100.0,
            )

        self.assertIn(
            "run_timestamp",
            summary,
        )

    # DATABASE I/O

    def test_load_from_db(self):
        """Data should be loaded correctly from a SQLite table."""

        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test.db"

            with sqlite3.connect(db_path) as conn:
                self.df.to_sql(
                    "transactions_clean",
                    conn,
                    index=False,
                )

            result = load_from_db(
                db_path,
                "transactions_clean",
            )

            self.assertEqual(
                len(result),
                3,
            )

            self.assertIn(
                "account",
                result.columns,
            )

            self.assertIn(
                "amount",
                result.columns,
            )

            self.assertEqual(
                result.loc[0, "account"],
                "1000",
            )

    # SUMMARY LOG

    def test_append_summary_log_creates_file(self):
        """A new summary log should be created if it does not exist."""

        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "logs" / "dq_history.csv"

            summary = {
                "run_timestamp": "2025-01-01T12:00:00",
                "total_rows": 3,
                "overall_score_pct": 100.0,
            }

            append_summary_log(
                summary,
                log_path,
            )

            self.assertTrue(log_path.exists())

            result = pd.read_csv(log_path)

            self.assertEqual(len(result), 1)
            self.assertEqual(result.loc[0, "total_rows"], 3)
            self.assertEqual(
                result.loc[0, "overall_score_pct"],
                100.0,
            )

    def test_append_summary_log_appends_to_existing_file(self):
        """A second summary should be appended without duplicating the header."""

        with TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "dq_history.csv"

            first_summary = {
                "run_timestamp": "2025-01-01T12:00:00",
                "total_rows": 3,
                "overall_score_pct": 100.0,
            }

            second_summary = {
                "run_timestamp": "2025-01-02T12:00:00",
                "total_rows": 5,
                "overall_score_pct": 90.0,
            }

            append_summary_log(
                first_summary,
                log_path,
            )

            append_summary_log(
                second_summary,
                log_path,
            )

            result = pd.read_csv(log_path)

            self.assertEqual(len(result), 2)
            self.assertEqual(result.loc[0, "total_rows"], 3)
            self.assertEqual(result.loc[1, "total_rows"], 5)

    # END-TO-END

    def test_run_data_quality(self):
        """The complete data-quality process should run successfully."""

        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "transactions.db"
            log_path = Path(tmp_dir) / "dq_history.csv"

            # Create the input SQLite table.
            with sqlite3.connect(db_path) as conn:
                self.df.to_sql(
                    "transactions_clean",
                    conn,
                    index=False,
                )

            summary = run_data_quality(
                db_path=db_path,
                table_name="transactions_clean",
                summary_log_path=log_path,
            )

            self.assertIsInstance(
                summary,
                dict,
            )

            self.assertEqual(
                summary["total_rows"],
                3,
            )

            self.assertEqual(
                summary["overall_score_pct"],
                100.0,
            )

            self.assertEqual(
                summary["rows_passing_all_checks_pct"],
                100.0,
            )

            self.assertTrue(
                log_path.exists()
            )

            log_df = pd.read_csv(log_path)

            self.assertEqual(
                len(log_df),
                1,
            )


if __name__ == "__main__":
    unittest.main()
