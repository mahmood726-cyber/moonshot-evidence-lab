from __future__ import annotations

import unittest
from pathlib import Path

from moonshot_evidence_lab.multivariate_meta import multivariate_meta_analysis
from moonshot_evidence_lab.network_meta import network_meta_analysis
from moonshot_evidence_lab.tabular_data import load_multivariate_wide_csv, load_network_contrast_csv


def examples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "examples"


class HigherOrderMetaTests(unittest.TestCase):
    def test_network_meta_returns_ranked_treatments(self) -> None:
        contrasts = load_network_contrast_csv(examples_dir() / "stroke_network_contrasts.csv")
        result = network_meta_analysis(
            contrasts,
            reference_treatment="Placebo",
            model="random",
        )
        self.assertEqual(result["reference_treatment"], "Placebo")
        self.assertEqual(result["treatment_count"], 4)
        self.assertEqual(len(result["ranking"]), 4)
        self.assertEqual(result["ranking"][0]["treatment"], "Cilostazol")
        self.assertLessEqual(float(result["residual_q_p_value"]), 1.0)

    def test_multivariate_meta_returns_outcome_vector(self) -> None:
        studies = load_multivariate_wide_csv(examples_dir() / "stroke_multivariate_wide.csv")
        result = multivariate_meta_analysis(studies, model="random", tau_method="pm")
        self.assertEqual(result["outcome_count"], 2)
        self.assertEqual(len(result["outcomes"]), 2)
        recurrence = next(item for item in result["outcomes"] if item["outcome"] == "stroke_recurrence")
        bleeding = next(item for item in result["outcomes"] if item["outcome"] == "major_bleeding")
        self.assertLess(float(recurrence["pooled_effect"]), 0.0)
        self.assertGreater(float(bleeding["pooled_effect"]), 0.0)
        self.assertEqual(len(result["pooled_covariance"]), 2)
        self.assertAlmostEqual(float(result["pooled_correlation"][0][0]), 1.0)

    def test_higher_order_loaders_parse_examples(self) -> None:
        contrasts = load_network_contrast_csv(examples_dir() / "stroke_network_contrasts.csv")
        studies = load_multivariate_wide_csv(examples_dir() / "stroke_multivariate_wide.csv")
        self.assertEqual(len(contrasts), 6)
        self.assertEqual(len(studies), 4)
        self.assertEqual(studies[0].outcomes, ["stroke_recurrence", "major_bleeding"])
        self.assertEqual(contrasts[0].metadata["metric"], "or")
        self.assertEqual(studies[0].metadata["effect_scale"], "log")


if __name__ == "__main__":
    unittest.main()
