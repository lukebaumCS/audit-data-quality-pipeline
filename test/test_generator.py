import unittest

from generator import GeneratorConfig


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


if __name__ == "__main__":
    unittest.main()
