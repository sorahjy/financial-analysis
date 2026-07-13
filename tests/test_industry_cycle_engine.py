import unittest

import pandas as pd

from industry_cycle_engine import (
    CycleForecastModel,
    IndustryCycleEngine,
    infer_cycle_phase,
)


def synthetic_plate(
    code,
    name,
    closes,
    *,
    turnover_base=1.0,
    amount_base=1.0,
    pe_base=20.0,
    pb_base=2.0,
):
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    rows = []
    for i, (date, close) in enumerate(zip(dates, closes)):
        rows.append(
            {
                "plate_code": code,
                "plate_name": name,
                "trade_date": date,
                "close_index": float(close),
                "change_pct": 0.0 if i == 0 else (float(close) / float(closes[i - 1]) - 1.0) * 100.0,
                "turnover_rate": turnover_base + i / max(len(closes), 1) * 0.3,
                "pe": pe_base + i / max(len(closes), 1),
                "pb": pb_base + i / max(len(closes), 1) * 0.1,
                "amount_share_pct": amount_base + i / max(len(closes), 1) * 0.2,
                "dividend_yield": 2.0,
                "source": "synthetic",
            }
        )
    return rows


class IndustryCycleEngineTest(unittest.TestCase):
    def test_infer_cycle_phase_marks_low_rebound_as_recovery(self):
        phase = infer_cycle_phase(
            18.0,
            ret20=4.5,
            ret60=-2.0,
            close=105.0,
            ma20=101.0,
            ma60=103.0,
        )

        self.assertEqual(phase, "recovery")

    def test_forecast_model_projects_mean_reverting_cycle_position(self):
        series = pd.Series([20 + min(i, 100) * 0.35 for i in range(220)])
        forecast = CycleForecastModel((20, 60)).predict(series)

        self.assertEqual(len(forecast), 2)
        self.assertIsNotNone(forecast[0].position)
        self.assertGreaterEqual(forecast[0].confidence, 5.0)
        self.assertLessEqual(forecast[0].position, 100.0)

    def test_build_payloads_from_frame_outputs_cycle_forecast_and_gated_smart_money(self):
        rows = []
        rows.extend(synthetic_plate("801001", "低位修复", list(range(150, 70, -1)) + list(range(70, 150))))
        rows.extend(synthetic_plate("801002", "高位行业", list(range(80, 240))))
        frame = pd.DataFrame(rows)

        engine = IndustryCycleEngine(min_history_days=120, forecast_horizons=(20, 60))
        payloads = engine.build_payloads_from_frame(frame)

        self.assertIn("records", payloads["cycle"])
        self.assertGreaterEqual(len(payloads["cycle"]["records"]), 2)
        first = payloads["cycle"]["records"][0]
        self.assertIn("forecast", first["meta"])
        self.assertEqual(first["meta"]["forecast"][0]["model"], "trend_mean_reversion_ensemble")

        smart_records = payloads["smart_money"]["records"]
        self.assertEqual(len(smart_records), len(payloads["cycle"]["records"]))
        self.assertTrue(all(record["only_valid_near_cycle_bottom"] for record in smart_records))

    def test_as_of_excludes_future_rows_from_percentiles_and_forecast(self):
        rows = []
        rows.extend(synthetic_plate("801001", "行业一", list(range(100, 320))))
        rows.extend(synthetic_plate("801002", "行业二", list(range(260, 40, -1))))
        frame = pd.DataFrame(rows)
        cutoff = pd.bdate_range("2024-01-01", periods=160)[-1].strftime("%Y-%m-%d")
        historical = frame[frame["trade_date"] <= pd.to_datetime(cutoff)].copy()

        engine = IndustryCycleEngine(min_history_days=120, forecast_horizons=(20, 60))
        expected = engine.build_payloads_from_frame(historical, as_of=cutoff)
        actual = engine.build_payloads_from_frame(frame, as_of=cutoff)

        self.assertEqual(actual["cycle"]["records"], expected["cycle"]["records"])
        self.assertEqual(actual["strength"]["records"], expected["strength"]["records"])
        self.assertEqual(actual["report"]["input_rows"], len(historical))


if __name__ == "__main__":
    unittest.main()
