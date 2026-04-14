from __future__ import annotations

import unittest
from pathlib import Path

from moonshot_evidence_lab.advanced_meta import (
    bayesian_random_effects,
    cumulative_meta_analysis,
    egger_test,
    format_advanced_summary,
    log_risk_ratio,
    meta_regression,
    multilevel_meta_analysis,
    standardized_mean_difference,
    subgroup_analysis,
    summarize_effects,
)
from moonshot_evidence_lab.tabular_data import load_effects_dataset


def examples_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "examples"


class AdvancedMetaTests(unittest.TestCase):
    def test_log_risk_ratio_returns_positive_variance(self) -> None:
        effect, variance = log_risk_ratio(12, 120, 20, 118)
        self.assertIsInstance(effect, float)
        self.assertGreater(variance, 0.0)

    def test_standardized_mean_difference_returns_positive_variance(self) -> None:
        effect, variance = standardized_mean_difference(0.18, 0.16, 81, 0.10, 0.18, 83)
        self.assertGreater(effect, 0.0)
        self.assertGreater(variance, 0.0)

    def test_binary_dataset_supports_advanced_summary(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "stroke_binary_trials.csv",
            dataset_format="binary",
            measure="or",
        )
        summary = summarize_effects(effects, model="random", ci_method="hartung-knapp", tau_method="pm")
        self.assertEqual(summary["k"], 6)
        self.assertLess(float(summary["ci_low"]), float(summary["ci_high"]))
        self.assertLessEqual(float(summary["q_p_value"]), 1.0)
        self.assertIsNotNone(summary["prediction_low"])
        self.assertIsNotNone(summary["prediction_high"])
        self.assertEqual(summary["tau_method"], "pm")
        self.assertGreaterEqual(float(summary["h"]), 1.0)

    def test_effects_dataset_supports_egger_and_subgroup(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "stroke_precomputed_effects.csv",
            dataset_format="effects",
        )
        egger = egger_test(effects)
        self.assertIsNotNone(egger)
        self.assertEqual(int(egger["k"]), 4)

        subgroup = subgroup_analysis(effects, "subgroup", model="random", ci_method="normal")
        self.assertEqual(subgroup["metadata_key"], "subgroup")
        self.assertEqual(subgroup["group_count"], 2)
        self.assertLessEqual(float(subgroup["p_value"]), 1.0)

    def test_continuous_dataset_supports_meta_regression(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "pulsatility_continuous_trials.csv",
            dataset_format="continuous",
            measure="md",
        )
        regression = meta_regression(effects, "dose_mg", model="random", tau_method="pm")
        self.assertIsNotNone(regression)
        self.assertEqual(regression["covariate_key"], "dose_mg")
        self.assertEqual(int(regression["k"]), 6)
        self.assertEqual(regression["tau_method"], "pm")

    def test_multivariable_meta_regression_returns_coefficients(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "stroke_multivariable_effects.csv",
            dataset_format="effects",
        )
        regression = meta_regression(
            effects,
            ["follow_up_months", "mean_age", "baseline_risk"],
            model="random",
            tau_method="pm",
        )
        self.assertIsNotNone(regression)
        self.assertEqual(regression["covariate_keys"], ["follow_up_months", "mean_age", "baseline_risk"])
        self.assertEqual(int(regression["moderator_df"]), 3)
        self.assertEqual(len(regression["coefficients"]), 4)

    def test_cumulative_meta_analysis_orders_by_year(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "stroke_multivariable_effects.csv",
            dataset_format="effects",
        )
        cumulative = cumulative_meta_analysis(
            effects,
            "year",
            model="random",
            ci_method="hartung-knapp",
            tau_method="pm",
        )
        self.assertEqual(len(cumulative), 6)
        self.assertEqual(cumulative[0]["order_value"], 2019)
        self.assertEqual(cumulative[-1]["k"], 6)

    def test_proportion_dataset_supports_summary_and_back_transform(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "hypertension_prevalence_surveys.csv",
            dataset_format="proportion",
            measure="logit",
        )
        summary = summarize_effects(effects, model="random", ci_method="hartung-knapp", tau_method="pm")
        formatted = format_advanced_summary(summary, effect_scale="logit")
        self.assertEqual(summary["k"], 6)
        self.assertGreater(float(formatted["pooled_estimate"]), 0.0)
        self.assertLess(float(formatted["pooled_estimate"]), 1.0)

    def test_bayesian_random_effects_returns_probabilities_and_intervals(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "stroke_binary_trials.csv",
            dataset_format="binary",
            measure="or",
        )
        posterior = bayesian_random_effects(
            effects,
            tau_prior="half-normal",
            tau_scale=0.2,
            grid_points=151,
        )
        self.assertEqual(posterior["model"], "bayesian-random-effects")
        self.assertEqual(int(posterior["k"]), 6)
        self.assertLess(float(posterior["ci_low"]), float(posterior["ci_high"]))
        self.assertGreaterEqual(float(posterior["posterior_probability_negative"]), 0.0)
        self.assertLessEqual(float(posterior["posterior_probability_negative"]), 1.0)
        self.assertGreaterEqual(float(posterior["tau_posterior_mean"]), 0.0)

    def test_multilevel_meta_analysis_returns_clustered_summary(self) -> None:
        effects = load_effects_dataset(
            examples_dir() / "stroke_clustered_effects.csv",
            dataset_format="effects",
        )
        summary = multilevel_meta_analysis(
            effects,
            "study_id",
            grid_points=11,
            refine_steps=2,
        )
        formatted = format_advanced_summary(summary, effect_scale="log")
        self.assertEqual(summary["model"], "multilevel-random-effects")
        self.assertEqual(int(summary["cluster_count"]), 3)
        self.assertEqual(len(summary["clusters"]), 3)
        self.assertLess(float(summary["ci_low"]), float(summary["ci_high"]))
        self.assertGreaterEqual(float(summary["tau_squared_between"]), 0.0)
        self.assertGreaterEqual(float(summary["tau_squared_within"]), 0.0)
        self.assertGreater(float(formatted["pooled_estimate"]), 0.0)

    def test_loaders_create_effects(self) -> None:
        binary_effects = load_effects_dataset(
            examples_dir() / "stroke_binary_trials.csv",
            dataset_format="binary",
            measure="rr",
        )
        continuous_effects = load_effects_dataset(
            examples_dir() / "pulsatility_continuous_trials.csv",
            dataset_format="continuous",
            measure="smd",
        )
        precomputed_effects = load_effects_dataset(
            examples_dir() / "stroke_precomputed_effects.csv",
            dataset_format="effects",
        )
        proportion_effects = load_effects_dataset(
            examples_dir() / "hypertension_prevalence_surveys.csv",
            dataset_format="proportion",
            measure="arcsine",
        )
        clustered_effects = load_effects_dataset(
            examples_dir() / "stroke_clustered_effects.csv",
            dataset_format="effects",
        )
        self.assertEqual(len(binary_effects), 6)
        self.assertEqual(len(continuous_effects), 6)
        self.assertEqual(len(precomputed_effects), 4)
        self.assertEqual(len(proportion_effects), 6)
        self.assertEqual(len(clustered_effects), 6)
        self.assertEqual(binary_effects[0].metadata["metric"], "rr")
        self.assertEqual(continuous_effects[0].metadata["metric"], "smd")
        self.assertEqual(proportion_effects[0].metadata["effect_scale"], "arcsine")


if __name__ == "__main__":
    unittest.main()
