import unittest

import pandas as pd

from src.generator import GeneratorConfig, TransactionGenerator


class TestGeneratorConfig(unittest.TestCase):

    def setUp(self):
        self.config = GeneratorConfig()

    def test_default_values(self):
        self.assertEqual(self.config.n_records, 100_000)
        self.assertEqual(self.config.anomaly_rate, 0.03)
        self.assertEqual(self.config.seed, 42)

    def test_date_range(self):
        self.assertLess(self.config.start_date, self.config.end_date)

    def test_currencies(self):
        self.assertIn("EUR", self.config.currencies)
        self.assertIn("USD", self.config.currencies)
        self.assertIn("JPY", self.config.currencies)
        self.assertEqual(len(self.config.currencies), 4)

    def test_accounts(self):
        self.assertEqual(len(self.config.accounts), 20)
        self.assertEqual(self.config.accounts[0], "4000")
        self.assertEqual(self.config.accounts[-1], "4095")

    def test_contra_accounts(self):
        self.assertEqual(len(self.config.contra_accounts), 20)
        self.assertEqual(self.config.contra_accounts[0], "1000")
        self.assertEqual(self.config.contra_accounts[-1], "1095")

    def test_cost_centers(self):
        self.assertEqual(len(self.config.cost_centers), 20)
        self.assertEqual(self.config.cost_centers[0], "CC-001")
        self.assertEqual(self.config.cost_centers[-1], "CC-020")

    def test_anomaly_weights(self):
        total = sum(self.config.anomaly_type_weights.values())

        self.assertAlmostEqual(total, 1.0)
        self.assertEqual(len(self.config.anomaly_type_weights), 6)

        for weight in self.config.anomaly_type_weights.values():
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, 1)


class TestTransactionGenerator(unittest.TestCase):

    def setUp(self):
        # Use a small dataset so tests execute quickly.
        self.config = GeneratorConfig(
            n_records=1_000,
            anomaly_rate=0.03,
            seed=42,
        )
        self.generator = TransactionGenerator(self.config)

    def test_generator_initialization(self):
        """Generator creates the expected clerks and clerk/account mapping."""
        self.assertEqual(
            len(self.generator.clerks),
            self.config.n_clerks,
        )

        self.assertEqual(
            len(self.generator._clerk_account_map),
            self.config.n_clerks,
        )

        for clerk, accounts in self.generator._clerk_account_map.items():
            self.assertIn(clerk, self.generator.clerks)
            self.assertGreaterEqual(len(accounts), 3)

    def test_generate_returns_dataframe(self):
        """generate() returns a pandas DataFrame."""
        df = self.generator.generate()

        self.assertIsInstance(df, pd.DataFrame)

    def test_generate_record_count_without_anomalies(self):
        """With anomaly_rate=0, exactly n_records are generated."""
        config = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator.generate()

        self.assertEqual(len(df), 500)

    def test_generate_record_count_with_duplicates(self):
        """
        Duplicate anomalies create additional rows.

        Therefore the final number of rows can be larger than n_records.
        """
        df = self.generator.generate()

        self.assertGreaterEqual(
            len(df),
            self.config.n_records,
        )

    def test_required_columns_exist(self):
        """Generated data contains all expected columns."""
        df = self.generator.generate()

        expected_columns = {
            "transaction_id",
            "document_number",
            "booking_datetime",
            "account",
            "contra_account",
            "amount",
            "currency",
            "cost_center",
            "clerk",
            "text",
            "payment_type",
            "is_anomaly",
            "anomaly_type",
        }

        self.assertTrue(
            expected_columns.issubset(set(df.columns))
        )

    def test_transaction_ids_are_present(self):
        """Every generated row has a transaction ID."""
        df = self.generator.generate()

        self.assertTrue(
            df["transaction_id"].notna().all()
        )

    def test_normal_transaction_ids_have_expected_format(self):
        """Normal transaction IDs follow TXN-XXXXXXXX."""
        config = GeneratorConfig(
            n_records=100,
            anomaly_rate=0.0,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        for transaction_id in df["transaction_id"]:
            self.assertRegex(
                transaction_id,
                r"^TXN-\d{8}$",
            )

    def test_amounts_are_positive(self):
        """Normal generated amounts should be positive."""
        config = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.0,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        self.assertTrue(
            (df["amount"] > 0).all()
        )

    def test_currency_values_are_valid(self):
        """Currency values must come from the configured currencies."""
        df = self.generator.generate()

        for currency in df["currency"].dropna():
            self.assertIn(
                currency,
                self.config.currencies,
            )

    def test_account_values_are_valid(self):
        """Account values must come from the configured accounts."""
        df = self.generator.generate()

        for account in df["account"].dropna():
            self.assertIn(
                account,
                self.config.accounts,
            )

    def test_contra_account_values_are_valid(self):
        """Contra accounts must come from the configured contra accounts."""
        df = self.generator.generate()

        for account in df["contra_account"].dropna():
            self.assertIn(
                account,
                self.config.contra_accounts,
            )

    def test_cost_center_values_are_valid(self):
        """Cost centres must come from the configured cost centres."""
        df = self.generator.generate()

        for cost_center in df["cost_center"].dropna():
            self.assertIn(
                cost_center,
                self.config.cost_centers,
            )

    def test_clerk_values_are_valid(self):
        """Clerks must come from the generated clerk list."""
        df = self.generator.generate()

        for clerk in df["clerk"].dropna():
            self.assertIn(
                clerk,
                self.generator.clerks,
            )

    def test_booking_datetime_is_datetime(self):
        """Booking datetime column contains datetime values."""
        config = GeneratorConfig(
            n_records=100,
            anomaly_rate=0.0,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                df["booking_datetime"]
            )
        )

    def test_normal_bookings_are_weekdays(self):
        """Normal transactions are generated Monday-Friday."""
        config = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.0,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        weekdays = df["booking_datetime"].dt.weekday

        self.assertTrue(
            (weekdays < 5).all()
        )

    def test_normal_bookings_are_business_hours(self):
        """Normal transactions are generated between 08:00 and 18:59."""
        config = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.0,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        hours = df["booking_datetime"].dt.hour

        self.assertTrue(
            ((hours >= 8) & (hours <= 18)).all()
        )

    def test_anomaly_columns_exist(self):
        """Anomaly metadata columns are created."""
        df = self.generator.generate()

        self.assertIn("is_anomaly", df.columns)
        self.assertIn("anomaly_type", df.columns)

    def test_anomaly_rate_is_approximately_correct(self):
        """
        The requested anomaly rate should be approximately represented.

        Duplicate anomalies mark both the original and duplicate row as
        anomalies, so the final rate can be slightly higher than configured.
        """
        config = GeneratorConfig(
            n_records=5_000,
            anomaly_rate=0.03,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        anomaly_rate = df["is_anomaly"].mean()

        # Allow some tolerance because anomalies are randomly distributed
        # and duplicates add an extra anomalous row.
        self.assertGreater(anomaly_rate, 0.02)
        self.assertLess(anomaly_rate, 0.06)

    def test_zero_anomaly_rate(self):
        """No anomalies are generated when anomaly_rate is zero."""
        config = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.0,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        self.assertEqual(
            df["is_anomaly"].sum(),
            0,
        )

        self.assertTrue(
            (df["anomaly_type"] == "").all()
        )

    def test_anomaly_types_are_valid(self):
        """Every anomaly type belongs to the configured anomaly types."""
        df = self.generator.generate()

        valid_types = set(
            self.config.anomaly_type_weights.keys()
        )
        valid_types.add("")

        for anomaly_type in df["anomaly_type"]:
            self.assertIn(
                anomaly_type,
                valid_types,
            )

    def test_high_amount_anomaly(self):
        """High amount anomaly increases the amount significantly."""
        config = GeneratorConfig(
            n_records=1,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(1)

        original_amount = df.loc[0, "amount"]

        generator._apply_anomaly(
            df,
            0,
            "high_amount",
        )

        self.assertGreater(
            df.loc[0, "amount"],
            original_amount,
        )

        self.assertTrue(
            df.loc[0, "amount"] >= original_amount * 8
        )

    def test_round_amount_anomaly(self):
        """Round amount anomaly produces one of the configured round values."""
        config = GeneratorConfig(
            n_records=1,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(1)

        generator._apply_anomaly(
            df,
            0,
            "round_amount",
        )

        allowed_amounts = {
            1000,
            5000,
            10000,
            25000,
            50000,
        }

        self.assertIn(
            df.loc[0, "amount"],
            allowed_amounts,
        )

    def test_weekend_night_posting_anomaly(self):
        """Weekend/night anomaly produces Saturday 00:00-04:59 or 23:00-23:59."""
        config = GeneratorConfig(
            n_records=1,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(1)

        generator._apply_anomaly(
            df,
            0,
            "weekend_night_posting",
        )

        dt = df.loc[0, "booking_datetime"]

        self.assertEqual(
            dt.weekday(),
            5,
        )

        self.assertIn(
            dt.hour,
            [0, 1, 2, 3, 4, 23],
        )

    def test_unusual_account_user_combo(self):
        """An unusual clerk/account combination uses an account outside the clerk's normal set."""
        config = GeneratorConfig(
            n_records=1,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(1)

        clerk = df.loc[0, "clerk"]
        original_account = df.loc[0, "account"]

        allowed_accounts = set(
            generator._clerk_account_map[clerk]
        )

        self.assertIn(
            original_account,
            allowed_accounts,
        )

        generator._apply_anomaly(
            df,
            0,
            "unusual_account_user_combo",
        )

        new_account = df.loc[0, "account"]

        self.assertNotIn(
            new_account,
            allowed_accounts,
        )

        self.assertIn(
            new_account,
            config.accounts,
        )

    def test_missing_field_anomaly(self):
        """Missing field anomaly sets one required field to None."""
        config = GeneratorConfig(
            n_records=1,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(1)

        generator._apply_anomaly(
            df,
            0,
            "missing_field",
        )

        self.assertTrue(
            df.loc[0, generator.REQUIRED_FIELDS].isna().any()
        )

        self.assertTrue(
            df.loc[0, "is_anomaly"]
        )

        self.assertEqual(
            df.loc[0, "anomaly_type"],
            "missing_field",
        )

    def test_duplicate_anomaly(self):
        """Duplicate anomaly creates an almost identical row and marks both rows."""
        config = GeneratorConfig(
            n_records=2,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(2)

        original_id = df.loc[0, "transaction_id"]
        original_document = df.loc[0, "document_number"]

        duplicate = generator._make_duplicate(
            df,
            0,
        )

        self.assertEqual(
            duplicate["transaction_id"],
            f"{original_id}-DUP",
        )

        self.assertEqual(
            duplicate["document_number"],
            f"{original_document}-DUP",
        )

        self.assertTrue(
            duplicate["is_anomaly"]
        )

        self.assertEqual(
            duplicate["anomaly_type"],
            "duplicate",
        )

        self.assertTrue(
            df.loc[0, "is_anomaly"]
        )

        self.assertEqual(
            df.loc[0, "anomaly_type"],
            "duplicate",
        )

    def test_duplicate_preserves_transaction_values(self):
        """A duplicate keeps the financial/business values of the original row."""
        config = GeneratorConfig(
            n_records=2,
            anomaly_rate=0.0,
            seed=42,
        )

        generator = TransactionGenerator(config)
        df = generator._generate_clean_records(2)

        duplicate = generator._make_duplicate(
            df,
            0,
        )

        fields_to_compare = [
            "booking_datetime",
            "account",
            "contra_account",
            "amount",
            "currency",
            "cost_center",
            "clerk",
            "text",
            "payment_type",
        ]

        for field in fields_to_compare:
            self.assertEqual(
                duplicate[field],
                df.loc[0, field],
            )

    def test_build_anomaly_plan_count(self):
        """Anomaly plan contains exactly the requested number of anomalies."""
        plan = self.generator._build_anomaly_plan(100)

        self.assertEqual(
            len(plan),
            100,
        )

    def test_build_anomaly_plan_contains_only_valid_types(self):
        """Anomaly plan only contains configured anomaly types."""
        plan = self.generator._build_anomaly_plan(500)

        valid_types = set(
            self.config.anomaly_type_weights.keys()
        )

        self.assertTrue(
            set(plan).issubset(valid_types)
        )

    def test_generate_is_deterministic(self):
        """Same seed and configuration produce identical data."""
        config1 = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.03,
            seed=123,
        )

        config2 = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.03,
            seed=123,
        )

        df1 = TransactionGenerator(config1).generate()
        df2 = TransactionGenerator(config2).generate()

        pd.testing.assert_frame_equal(
            df1,
            df2,
        )

    def test_different_seed_produces_different_data(self):
        """Different seeds should normally produce different datasets."""
        config1 = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.03,
            seed=123,
        )

        config2 = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.03,
            seed=456,
        )

        df1 = TransactionGenerator(config1).generate()
        df2 = TransactionGenerator(config2).generate()

        self.assertFalse(
            df1.equals(df2)
        )

    def test_transaction_ids_are_unique(self):
        """
        Transaction IDs remain unique, including duplicate rows,
        because duplicates receive a -DUP suffix.
        """
        df = self.generator.generate()

        self.assertEqual(
            df["transaction_id"].nunique(),
            len(df),
        )

    def test_duplicate_rows_are_marked_correctly(self):
        """Duplicate anomalies mark both the original and duplicate row as anomalies."""
        config = GeneratorConfig(
            n_records=500,
            anomaly_rate=0.05,
            seed=42,
        )

        df = TransactionGenerator(config).generate()

        duplicates = df[
            df["anomaly_type"] == "duplicate"
        ]

        if not duplicates.empty:
            # Both the original and duplicate are marked as anomalies.
            self.assertTrue(
                duplicates["is_anomaly"].all()
            )

            # At least one generated duplicate must have the -DUP suffix.
            duplicate_rows = duplicates[
                duplicates["transaction_id"].str.endswith("-DUP")
            ]

            self.assertGreater(
                len(duplicate_rows),
                0,
        )

if __name__ == "__main__":
    unittest.main()