import pandas as pd
import pytest

from src.anomaly_detection import (
    AnomalyDetectionConfig,
    RuleBasedAnomalyDetector,
    RULE_NAMES,
)


@pytest.fixture
def detector():
    return RuleBasedAnomalyDetector()


@pytest.fixture
def custom_detector():
    config = AnomalyDetectionConfig(
        business_hour_start=8,
        business_hour_end=18,
        round_amount_threshold=500.0,
        round_amount_divisor=500.0,
        high_amount_quantile=0.99,
        unusual_combo_min_frequency=0.02,
    )
    return RuleBasedAnomalyDetector(config)


# ---------------------------------------------------------------------------
# detect_duplicates
# ---------------------------------------------------------------------------

def test_detect_duplicates_uses_existing_duplicate_column(detector):
    df = pd.DataFrame({
        "account": [1000, 2000],
        "is_potential_duplicate": [True, False],
    })

    result = detector.detect_duplicates(df)

    assert result.tolist() == [True, False]
    assert result.dtype == bool


def test_detect_duplicates_detects_duplicate_rows(detector):
    df = pd.DataFrame({
        "account": [1000, 1000, 2000],
        "contra_account": [2000, 2000, 3000],
        "amount": [100.0, 100.0, 200.0],
        "cost_center": ["A", "A", "B"],
        "clerk": ["Müller", "Müller", "Schmidt"],
        "booking_date": ["2025-01-01", "2025-01-01", "2025-01-02"],
    })

    result = detector.detect_duplicates(df)

    assert result.tolist() == [True, True, False]


def test_detect_duplicates_does_not_flag_unique_rows(detector):
    df = pd.DataFrame({
        "account": [1000, 1000],
        "contra_account": [2000, 2000],
        "amount": [100.0, 101.0],
        "cost_center": ["A", "A"],
        "clerk": ["Müller", "Müller"],
        "booking_date": ["2025-01-01", "2025-01-01"],
    })

    result = detector.detect_duplicates(df)

    assert result.tolist() == [False, False]


# ---------------------------------------------------------------------------
# detect_weekend_night
# ---------------------------------------------------------------------------

def test_detect_weekend_night_flags_night(detector):
    df = pd.DataFrame({
        "booking_datetime": [
            "2025-01-06 05:59:00",
            "2025-01-06 06:00:00",
            "2025-01-06 19:59:00",
            "2025-01-06 21:00:00",
        ]
    })

    result = detector.detect_weekend_night(df)

    assert result.tolist() == [True, False, False, True]


def test_detect_weekend_night_flags_night(detector):
    df = pd.DataFrame({
        "booking_datetime": [
            "2025-01-06 05:59:00",
            "2025-01-06 06:00:00",
            "2025-01-06 20:00:00",
            "2025-01-06 21:00:00",
        ]
    })

    result = detector.detect_weekend_night(df)

    assert result.tolist() == [True, False, False, True]


def test_detect_weekend_night_handles_invalid_datetime(detector):
    df = pd.DataFrame({
        "booking_datetime": [
            "not-a-date",
            None,
            "2025-01-06 12:00:00",
        ]
    })

    result = detector.detect_weekend_night(df)

    assert result.tolist() == [False, False, False]


# ---------------------------------------------------------------------------
# detect_round_amount
# ---------------------------------------------------------------------------

def test_detect_round_amount_flags_large_round_amounts(detector):
    df = pd.DataFrame({
        "amount": [
            499.99,
            500.0,
            1000.0,
            1500.0,
            1501.0,
        ]
    })

    result = detector.detect_round_amount(df)

    assert result.tolist() == [
        False,  # unter threshold
        True,   # 500
        True,   # 1000
        True,   # 1500
        False,  # nicht durch 500 teilbar
    ]


def test_detect_round_amount_handles_invalid_amount(detector):
    df = pd.DataFrame({
        "amount": [
            "abc",
            None,
            500.0,
        ]
    })

    result = detector.detect_round_amount(df)

    assert result.tolist() == [False, False, True]


def test_detect_round_amount_respects_custom_config():
    config = AnomalyDetectionConfig(
        round_amount_threshold=1000.0,
        round_amount_divisor=100.0,
    )
    detector = RuleBasedAnomalyDetector(config)

    df = pd.DataFrame({
        "amount": [500, 900, 1000, 1100, 1150]
    })

    result = detector.detect_round_amount(df)

    assert result.tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]


# ---------------------------------------------------------------------------
# detect_high_amount
# ---------------------------------------------------------------------------

def test_detect_high_amount_flags_values_above_quantile(detector):
    # Bei Quantile 0.99 wird nur der Wert oberhalb des 99%-Quantils
    # als Anomalie markiert.
    amounts = list(range(1, 101))

    df = pd.DataFrame({
        "amount": amounts
    })

    result = detector.detect_high_amount(df)

    assert result.sum() == 1
    assert bool(result.iloc[-1])


def test_detect_high_amount_does_not_flag_values_at_quantile(detector):
    df = pd.DataFrame({
        "amount": [100, 200, 300, 400, 500]
    })

    config = AnomalyDetectionConfig(high_amount_quantile=0.8)
    detector = RuleBasedAnomalyDetector(config)

    result = detector.detect_high_amount(df)

    threshold = pd.Series(df["amount"]).quantile(0.8)

    assert not result[df["amount"] <= threshold].any()


def test_detect_high_amount_handles_invalid_values(detector):
    df = pd.DataFrame({
        "amount": [100, 200, "invalid", None, 1000]
    })

    result = detector.detect_high_amount(df)

    assert bool(result.iloc[2]) is False
    assert bool(result.iloc[3]) is False
    assert bool(result.iloc[4]) is True


# ---------------------------------------------------------------------------
# detect_unusual_account_combo
# ---------------------------------------------------------------------------

def test_detect_unusual_account_combo_flags_rare_clerk_account_combination():
    config = AnomalyDetectionConfig(
        unusual_combo_min_frequency=0.2
    )
    detector = RuleBasedAnomalyDetector(config)

    # Müller:
    # account A -> 4 / 5 = 80 %
    # account B -> 1 / 5 = 20 %
    #
    # Mit "< 0.2" wird nichts geflaggt, weil B genau 20 % hat.
    df = pd.DataFrame({
        "clerk": ["Müller"] * 5,
        "account": ["A", "A", "A", "A", "B"],
    })

    result = detector.detect_unusual_account_combo(df)

    assert result.tolist() == [False, False, False, False, False]


def test_detect_unusual_account_combo_flags_below_threshold():
    config = AnomalyDetectionConfig(
        unusual_combo_min_frequency=0.2
    )
    detector = RuleBasedAnomalyDetector(config)

    # Müller:
    # A -> 4 / 6 = 66.7 %
    # B -> 1 / 6 = 16.7 %
    # C -> 1 / 6 = 16.7 %
    df = pd.DataFrame({
        "clerk": ["Müller"] * 6,
        "account": ["A", "A", "A", "A", "B", "C"],
    })

    result = detector.detect_unusual_account_combo(df)

    assert result.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_detect_unusual_account_combo_is_calculated_per_clerk(detector):
    config = AnomalyDetectionConfig(
        unusual_combo_min_frequency=0.5
    )
    detector = RuleBasedAnomalyDetector(config)

    df = pd.DataFrame({
        "clerk": [
            "Müller",
            "Müller",
            "Müller",
            "Müller",
            "Schmidt",
            "Schmidt",
            "Schmidt",
            "Schmidt",
        ],
        "account": [
            "A",
            "A",
            "A",
            "B",
            "A",
            "A",
            "B",
            "B",
        ],
    })

    result = detector.detect_unusual_account_combo(df)

    # Müller:
    # A = 3/4 = 75 %, B = 1/4 = 25 %
    #
    # Schmidt:
    # A = 2/4 = 50 %, B = 2/4 = 50 %
    #
    # Nur B bei Müller ist < 50 %.
    assert result.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_creates_all_rule_columns(detector):
    df = pd.DataFrame({
        "account": [1000, 1000, 2000],
        "contra_account": [2000, 2000, 3000],
        "amount": [500.0, 500.0, 100.0],
        "cost_center": ["A", "A", "B"],
        "clerk": ["Müller", "Müller", "Schmidt"],
        "booking_date": [
            "2025-01-01",
            "2025-01-01",
            "2025-01-02",
        ],
        "booking_datetime": [
            "2025-01-06 12:00:00",
            "2025-01-06 12:00:00",
            "2025-01-06 12:00:00",
        ],
    })

    result = detector.run(df)

    for rule_name in RULE_NAMES:
        assert rule_name in result.columns

    assert "rules_triggered_count" in result.columns
    assert "is_flagged" in result.columns
    assert "triggered_rules" in result.columns


def test_run_does_not_modify_original_dataframe(detector):
    df = pd.DataFrame({
        "account": [1000],
        "contra_account": [2000],
        "amount": [100.0],
        "cost_center": ["A"],
        "clerk": ["Müller"],
        "booking_date": ["2025-01-01"],
        "booking_datetime": ["2025-01-06 12:00:00"],
    })

    original_columns = df.columns.tolist()

    result = detector.run(df)

    assert df.columns.tolist() == original_columns
    assert "is_flagged" not in df.columns
    assert "is_flagged" in result.columns


def test_run_calculates_triggered_rules(detector):
    df = pd.DataFrame({
        "account": [1000, 2000],
        "contra_account": [2000, 3000],
        "amount": [500.0, 100.0],
        "cost_center": ["A", "B"],
        "clerk": ["Müller", "Schmidt"],
        "booking_date": ["2025-01-01", "2025-01-02"],
        "booking_datetime": [
            "2025-01-06 12:00:00",
            "2025-01-06 12:00:00",
        ],
    })

    result = detector.run(df)

    assert result["rule_round_amount"].tolist() == [True, False]
    assert result["triggered_rules"].iloc[0].split(",").count("round_amount") == 1
    assert bool(result["is_flagged"].iloc[0])


def test_run_rules_triggered_count_matches_rule_columns(detector):
    df = pd.DataFrame({
        "account": [1000, 2000],
        "contra_account": [2000, 3000],
        "amount": [500.0, 100.0],
        "cost_center": ["A", "B"],
        "clerk": ["Müller", "Schmidt"],
        "booking_date": ["2025-01-01", "2025-01-02"],
        "booking_datetime": [
            "2025-01-06 12:00:00",
            "2025-01-06 12:00:00",
        ],
    })

    result = detector.run(df)

    expected_count = result[RULE_NAMES].sum(axis=1)

    pd.testing.assert_series_equal(
        result["rules_triggered_count"],
        expected_count,
        check_names=False,
    )


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def test_summarize_returns_correct_values(detector):
    df = pd.DataFrame({
        "is_flagged": [True, True, False, False],
        "rule_duplicate": [True, False, False, False],
        "rule_weekend_night": [False, True, False, False],
        "rule_round_amount": [True, False, False, False],
        "rule_high_amount": [False, True, False, False],
        "rule_unusual_account_combo": [False, False, True, False],
    })

    result = detector.summarize(df)

    assert result["total_rows"] == 4
    assert result["flagged_rows"] == 2
    assert result["flagged_rate_pct"] == 50.0

    assert result["rule_duplicate_count"] == 1
    assert result["rule_weekend_night_count"] == 1
    assert result["rule_round_amount_count"] == 1
    assert result["rule_high_amount_count"] == 1
    assert result["rule_unusual_account_combo_count"] == 1


def test_summarize_empty_dataframe(detector):
    df = pd.DataFrame({
        "is_flagged": pd.Series(dtype=bool),
        "rule_duplicate": pd.Series(dtype=bool),
        "rule_weekend_night": pd.Series(dtype=bool),
        "rule_round_amount": pd.Series(dtype=bool),
        "rule_high_amount": pd.Series(dtype=bool),
        "rule_unusual_account_combo": pd.Series(dtype=bool),
    })

    result = detector.summarize(df)

    assert result["total_rows"] == 0
    assert result["flagged_rows"] == 0
    # pandas mean() bei leerem DataFrame ist NaN.
    assert pd.isna(result["flagged_rate_pct"])


# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

def test_default_config_is_used():
    detector = RuleBasedAnomalyDetector()

    assert detector.config.business_hour_start == 6
    assert detector.config.business_hour_end == 20
    assert detector.config.round_amount_threshold == 500.0
    assert detector.config.round_amount_divisor == 500.0
    assert detector.config.high_amount_quantile == 0.99
    assert detector.config.unusual_combo_min_frequency == 0.02


def test_custom_config_is_used():
    config = AnomalyDetectionConfig(
        business_hour_start=8,
        business_hour_end=17,
        round_amount_threshold=1000.0,
    )

    detector = RuleBasedAnomalyDetector(config)

    assert detector.config.business_hour_start == 8
    assert detector.config.business_hour_end == 17
    assert detector.config.round_amount_threshold == 1000.0