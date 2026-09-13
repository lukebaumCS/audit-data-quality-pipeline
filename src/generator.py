from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker



### Config Class

@dataclass
class GeneratorConfig:
    n_records: int = 100_000
    anomaly_rate: float = 0.03                          # Proportion of rows with an injected anomaly
    seed: int = 42

    start_date: datetime = datetime(2004, 6, 14)
    end_date: datetime = datetime(2026, 6, 14)

    currencies: tuple[str, ...] = ("EUR", "USD", "JPY", "GBP")
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


### Generator

class TransactionGenerator:

    REQUIRED_FIELDS = ["account", "amount", "cost_center", "clerk", "booking_datetime"]

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.faker = Faker("de_DE")
        Faker.seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.clerks = [self.faker.name() for _ in range(self.config.n_clerks)]
        # Simulates a "normal" assignment: each clerk usually posts only
        # to a subset of the accounts -> basis for "unusual_account_user_combo".
        self._clerk_account_map = {
            clerk: random.sample(self.config.accounts, k=max(3, len(self.config.accounts) // 3))
            for clerk in self.clerks
        }

    # Public API 

    def generate(self) -> pd.DataFrame:
        df = self._generate_clean_records(self.config.n_records)
        df["is_anomaly"] = False
        df["anomaly_type"] = ""

        n_anomalies = int(self.config.n_records * self.config.anomaly_rate)
        anomaly_plan = self._build_anomaly_plan(n_anomalies)

        extra_rows = []
        target_indices = random.sample(range(len(df)), k=len(anomaly_plan))

        for idx, anomaly_type in zip(target_indices, anomaly_plan):
            if anomaly_type == "duplicate":
                extra_rows.append(self._make_duplicate(df, idx))
            else:
                self._apply_anomaly(df, idx, anomaly_type)

        if extra_rows:
            df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

        return df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)

    def save(self, df: pd.DataFrame, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    # Internal helper functions 

    def _generate_clean_records(self, n: int) -> pd.DataFrame:
        cfg = self.config
        rows = []
        for i in range(n):
            clerk = random.choice(self.clerks)
            account = random.choice(self._clerk_account_map[clerk])
            rows.append({
                "transaction_id": f"TXN-{i:08d}",
                "document_number": f"BELEG-{self.faker.unique.random_number(digits=8)}",
                "booking_datetime": self._random_business_datetime(),
                "account": account,
                "contra_account": random.choice(cfg.contra_accounts),
                "amount": round(np.random.lognormal(mean=6.5, sigma=1.0), 2),
                "currency": random.choices(cfg.currencies, weights=[00.80, 0.10, 0.05, 0.05])[0],
                "cost_center": random.choice(cfg.cost_centers),
                "clerk": clerk,
                "text": self.faker.bs(),
                "payment_type": random.choice(cfg.payment_types),
            })
        return pd.DataFrame(rows)

    def _random_business_datetime(self) -> datetime:
        """Date/time on weekdays, 08:00–18:00 (under normal circumstances, with no anomalies)."""        
        
        cfg = self.config
        delta_days = (cfg.end_date - cfg.start_date).days
        while True:
            dt = cfg.start_date + timedelta(
                days=random.randint(0, delta_days),
                hours=random.randint(8, 18),
                minutes=random.randint(0, 59),
            )
            if dt.weekday() < 5:  # Monday–Friday
                return dt

    def _build_anomaly_plan(self, n_anomalies: int) -> list[str]:
        weights = self.config.anomaly_type_weights
        types = list(weights.keys())
        probs = list(weights.values())
        return list(np.random.choice(types, size=n_anomalies, p=probs))

    def _make_duplicate(self, df: pd.DataFrame, idx: int) -> dict:
        """Copies an existing row almost unchanged (a classic example of a duplicate entry)."""
        original = df.loc[idx].to_dict()
        duplicate = original.copy()
        duplicate["transaction_id"] = f"{original['transaction_id']}-DUP"
        duplicate["document_number"] = f"{original['document_number']}-DUP"
        duplicate["is_anomaly"] = True
        duplicate["anomaly_type"] = "duplicate"
        
        # Mark the original as part of the duplicate pair as well:
        df.loc[idx, "is_anomaly"] = True
        df.loc[idx, "anomaly_type"] = "duplicate"
        return duplicate

    def _apply_anomaly(self, df: pd.DataFrame, idx: int, anomaly_type: str) -> None:
        if anomaly_type == "high_amount":
            factor = random.uniform(8, 25)
            df.loc[idx, "amount"] = round(df.loc[idx, "amount"] * factor, 2)

        elif anomaly_type == "round_amount":
            df.loc[idx, "amount"] = float(random.choice([1000, 5000, 10000, 25000, 50000]))

        elif anomaly_type == "weekend_night_posting":
            base_date = self.faker.date_between(
                start_date=self.config.start_date, end_date=self.config.end_date
            )
            # Postpone it until the weekend and schedule it for the evening
            days_to_saturday = (5 - base_date.weekday()) % 7
            weekend_date = base_date + timedelta(days=days_to_saturday)
            night_hour = random.choice([0, 1, 2, 3, 4, 23])
            df.loc[idx, "booking_datetime"] = datetime.combine(
                weekend_date, datetime.min.time()
            ) + timedelta(hours=night_hour, minutes=random.randint(0, 59))

        elif anomaly_type == "unusual_account_user_combo":
            clerk = df.loc[idx, "clerk"]
            allowed = set(self._clerk_account_map[clerk])
            unusual_choices = [a for a in self.config.accounts if a not in allowed]
            if unusual_choices:
                df.loc[idx, "account"] = random.choice(unusual_choices)

        elif anomaly_type == "missing_field":
            field_name = random.choice(self.REQUIRED_FIELDS)
            df.loc[idx, field_name] = None

        df.loc[idx, "is_anomaly"] = True
        df.loc[idx, "anomaly_type"] = anomaly_type



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetischer Journal-Entry-Generator")
    parser.add_argument("--n-records", type=int, default=100_000)
    parser.add_argument("--anomaly-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = GeneratorConfig(
        n_records=args.n_records,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    generator = TransactionGenerator(config)
    df = generator.generate()
    generator.save(df, args.output)

    print(f"{len(df):,} Zeilen erzeugt -> {args.output}")
    print(f"Davon Anomalien: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean():.2%})")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()

