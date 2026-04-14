from __future__ import annotations

import unittest

from moonshot_evidence_lab.meta_analysis import estimate_tau_squared, log_odds_ratio, run_meta_analysis
from moonshot_evidence_lab.models import StudyEffect


class MetaAnalysisTests(unittest.TestCase):
    def test_log_odds_ratio_returns_positive_variance(self) -> None:
        effect, variance = log_odds_ratio(12, 88, 20, 80)
        self.assertIsInstance(effect, float)
        self.assertGreater(variance, 0)

    def test_random_effects_meta_analysis_runs(self) -> None:
        effects = [
            StudyEffect("s1", "s1", -0.20, 0.04),
            StudyEffect("s2", "s2", -0.10, 0.03),
            StudyEffect("s3", "s3", -0.16, 0.05),
        ]
        result = run_meta_analysis(effects, model="random")
        self.assertEqual(result.k, 3)
        self.assertLess(result.ci_low, result.ci_high)
        self.assertGreaterEqual(result.i_squared, 0.0)
        self.assertGreaterEqual(result.h, 1.0)
        self.assertGreaterEqual(result.h_squared, 1.0)

    def test_paule_mandel_tau_estimator_runs(self) -> None:
        effects = [
            StudyEffect("s1", "s1", -0.42, 0.08),
            StudyEffect("s2", "s2", -0.05, 0.05),
            StudyEffect("s3", "s3", -0.31, 0.06),
            StudyEffect("s4", "s4", 0.02, 0.07),
        ]
        tau_squared = estimate_tau_squared(effects, tau_method="pm")
        result = run_meta_analysis(effects, model="random", tau_method="pm")
        self.assertGreaterEqual(tau_squared, 0.0)
        self.assertEqual(result.tau_method, "pm")
        self.assertGreaterEqual(result.tau_squared, 0.0)


if __name__ == "__main__":
    unittest.main()
