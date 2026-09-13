from dataclasses import dataclass, field
from datetime import datetime

### config class

@dataclass
class GeneratorConfig:
    n_records: int = 100_000
    anomaly_rate: float = 0.03                          # Proportion of rows with an injected anomaly
    seed: int = 42

    start_date: datetime = datetime(2004, 6, 14)
    end_date: datetime = datetime(2026, 6, 14)

    currencies: tuple[str, ...] = ("EUR", "USD", "YEN", "GBP")
    accounts: tuple[str, ...] = tuple(str(a) for a in range(4000, 4100, 5))         # 20 Expense account
    contra_accounts: tuple[str, ...] = tuple(str(a) for a in range(1000, 1100, 5))  # 20 Contra account
    cost_centers: tuple[str, ...] = tuple(f"CC-{i:03d}" for i in range(1, 21))      # 20 Cost centres
    payment_types: tuple[str, ...] = ("Überweisung", "Lastschrift", "Scheck", "Barzahlung", "Kreditkarte", "PayPal")

    n_clerks: int = 25
    
    # Proportion of anomaly rows attributable to each anomaly type.
    # Must add up to 1.0.
    anomaly_type_weights: dict[str, float] = field(default_factory=lambda: {
        "duplicate": 0.20,
        "high_amount": 0.15,
        "round_amount": 0.15,
        "weekend_night_posting": 0.20,
        "unusual_account_user_combo": 0.20,
        "missing_field": 0.10,
    })


def main() -> None:
    config = GeneratorConfig()

if __name__ == "__main__":
    main()
