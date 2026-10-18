from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .advanced_meta import (
    back_transform,
    bayesian_random_effects,
    cumulative_meta_analysis,
    egger_test,
    format_advanced_summary,
    leave_one_out_analysis,
    meta_regression,
    multilevel_meta_analysis,
    subgroup_analysis,
    summarize_effects,
)
from .autoresearch_loop import demo_effects, run_demo_benchmark
from .benchmarks import run_literature_screening_benchmark
from .connectors.clinicaltrials import ClinicalTrialsGovClient
from .connectors.ihme import default_template_path, write_template_catalog
from .connectors.pubmed import PubMedClient
from .connectors.who import WHOClient
from .extraction import extract_ctgov_effect_candidates, extract_effect_candidates_from_text
from .meta_analysis import log_odds_ratio, odds_ratio_from_log_effect, run_meta_analysis
from .multivariate_meta import multivariate_meta_analysis
from .models import StudyEffect
from .network_meta import network_meta_analysis
from .review_pipeline import run_review_pipeline
from .tabular_data import load_effects_dataset, load_multivariate_wide_csv, load_network_contrast_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moonshot-evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke-test", help="Run the local benchmark and connector smoke test")
    subparsers.add_parser("meta-demo", help="Run a demo meta-analysis from 2x2 counts")
    benchmark_parser = subparsers.add_parser("benchmark", help="Print the demo benchmark scorecard")
    benchmark_parser.set_defaults(command="benchmark")
    subparsers.add_parser(
        "literature-benchmark",
        help="Score the literature-screening benchmark against the current program surface",
    )

    who_parser = subparsers.add_parser("who-url", help="Build a WHO indicator URL")
    who_parser.add_argument("--indicator", required=True, help="WHO indicator code")
    who_parser.add_argument("--top", type=int, default=100, help="Number of rows to request")
    who_parser.add_argument("--filter", default="", help="Optional OData filter expression")

    pubmed_parser = subparsers.add_parser("pubmed-url", help="Build a PubMed ESearch URL")
    pubmed_parser.add_argument("--term", required=True, help="PubMed query")
    pubmed_parser.add_argument("--retmax", type=int, default=20, help="Number of IDs to request")
    pubmed_parser.add_argument("--sort", default="relevance", help="NCBI sort order")

    pubmed_search_parser = subparsers.add_parser("pubmed-search", help="Search PubMed and normalize summaries")
    pubmed_search_parser.add_argument("--term", required=True, help="PubMed query")
    pubmed_search_parser.add_argument("--retmax", type=int, default=10, help="Number of records to request")
    pubmed_search_parser.add_argument("--sort", default="relevance", help="NCBI sort order")

    pubmed_extract_parser = subparsers.add_parser(
        "pubmed-extract",
        help="Fetch a PubMed abstract and extract effect-size candidates",
    )
    pubmed_extract_parser.add_argument("--pmid", required=True, help="PubMed ID")

    ctgov_parser = subparsers.add_parser("ctgov-url", help="Build a ClinicalTrials.gov search URL")
    ctgov_parser.add_argument("--term", default="", help="Basic search query")
    ctgov_parser.add_argument("--condition", default="", help="Condition search query")
    ctgov_parser.add_argument("--intervention", default="", help="Intervention search query")
    ctgov_parser.add_argument("--page-size", type=int, default=20, help="Requested page size")

    ctgov_search_parser = subparsers.add_parser(
        "ctgov-search",
        help="Search ClinicalTrials.gov and normalize study records",
    )
    ctgov_search_parser.add_argument("--term", default="", help="Basic search query")
    ctgov_search_parser.add_argument("--condition", default="", help="Condition search query")
    ctgov_search_parser.add_argument("--intervention", default="", help="Intervention search query")
    ctgov_search_parser.add_argument("--page-size", type=int, default=10, help="Requested page size")

    ctgov_extract_parser = subparsers.add_parser(
        "ctgov-extract",
        help="Fetch a ClinicalTrials.gov study and extract effect-size candidates from results",
    )
    ctgov_extract_parser.add_argument("--nct-id", required=True, help="ClinicalTrials.gov NCT ID")

    review_parser = subparsers.add_parser(
        "review-pipeline",
        help="Run search, extraction, and pooling for a review question",
    )
    review_parser.add_argument("--term", default="", help="Primary search term")
    review_parser.add_argument("--condition", default="", help="Optional ClinicalTrials.gov condition")
    review_parser.add_argument("--intervention", default="", help="Optional ClinicalTrials.gov intervention")
    review_parser.add_argument("--pubmed-retmax", type=int, default=5, help="PubMed records to request")
    review_parser.add_argument("--ctgov-page-size", type=int, default=5, help="ClinicalTrials.gov records per page; all pages are collected")
    review_parser.add_argument("--min-studies", type=int, default=2, help="Minimum studies required to pool a group")
    review_parser.add_argument("--pubmed-pmid", action="append", default=[], help="Seed the pipeline with a specific PMID")
    review_parser.add_argument("--ctgov-nct-id", action="append", default=[], help="Seed the pipeline with a specific NCT ID")
    review_parser.add_argument(
        "--include-secondary-evidence",
        action="store_true",
        help="Allow review or meta-analysis articles from PubMed into pooled effects",
    )

    advanced_parser = subparsers.add_parser(
        "advanced-meta",
        help="Run advanced meta-analysis diagnostics on a local CSV dataset",
    )
    advanced_parser.add_argument("--input", required=True, help="CSV file path")
    advanced_parser.add_argument(
        "--format",
        required=True,
        choices=["effects", "binary", "continuous", "proportion"],
        help="Dataset format",
    )
    advanced_parser.add_argument(
        "--measure",
        default="",
        help="Effect measure: or/rr/rd for binary, md/smd/rom for continuous, or logit/raw/arcsine for proportion",
    )
    advanced_parser.add_argument(
        "--model",
        default="random",
        choices=["fixed", "random"],
        help="Meta-analysis model",
    )
    advanced_parser.add_argument(
        "--tau-method",
        default="pm",
        choices=["dl", "pm"],
        help="Between-study variance estimator for random-effects analyses",
    )
    advanced_parser.add_argument(
        "--ci-method",
        default="hartung-knapp",
        choices=["normal", "hartung-knapp"],
        help="Confidence interval method",
    )
    advanced_parser.add_argument("--outcome", default="", help="Filter to a single outcome label")
    advanced_parser.add_argument("--subgroup-key", default="", help="Metadata key for subgroup analysis")
    advanced_parser.add_argument(
        "--meta-regression-key",
        action="append",
        default=[],
        help="Metadata key for meta-regression; repeat for multiple covariates",
    )
    advanced_parser.add_argument("--cumulative-key", default="", help="Metadata key for cumulative meta-analysis ordering")
    advanced_parser.add_argument(
        "--skip-leave-one-out",
        action="store_true",
        help="Disable leave-one-out diagnostics",
    )

    bayesian_parser = subparsers.add_parser(
        "bayesian-meta",
        help="Run Bayesian random-effects pooling on a local CSV dataset",
    )
    bayesian_parser.add_argument("--input", required=True, help="CSV file path")
    bayesian_parser.add_argument(
        "--format",
        required=True,
        choices=["effects", "binary", "continuous", "proportion"],
        help="Dataset format",
    )
    bayesian_parser.add_argument(
        "--measure",
        default="",
        help="Effect measure: or/rr/rd for binary, md/smd/rom for continuous, or logit/raw/arcsine for proportion",
    )
    bayesian_parser.add_argument("--outcome", default="", help="Filter to a single outcome label")
    bayesian_parser.add_argument(
        "--tau-prior",
        default="half-normal",
        choices=["half-normal", "half-cauchy", "uniform"],
        help="Prior family for tau",
    )
    bayesian_parser.add_argument("--tau-scale", type=float, default=0.2, help="Scale parameter for the tau prior")
    bayesian_parser.add_argument("--mu-prior-mean", type=float, default=0.0, help="Prior mean for the pooled effect")
    bayesian_parser.add_argument("--mu-prior-sd", type=float, default=1.0, help="Prior standard deviation for the pooled effect")
    bayesian_parser.add_argument("--grid-points", type=int, default=301, help="Number of tau grid points")

    multilevel_parser = subparsers.add_parser(
        "multilevel-meta",
        help="Run clustered multilevel random-effects pooling on a local CSV dataset",
    )
    multilevel_parser.add_argument("--input", required=True, help="CSV file path")
    multilevel_parser.add_argument(
        "--format",
        required=True,
        choices=["effects", "binary", "continuous", "proportion"],
        help="Dataset format",
    )
    multilevel_parser.add_argument(
        "--measure",
        default="",
        help="Effect measure: or/rr/rd for binary, md/smd/rom for continuous, or logit/raw/arcsine for proportion",
    )
    multilevel_parser.add_argument("--outcome", default="", help="Filter to a single outcome label")
    multilevel_parser.add_argument(
        "--cluster-key",
        default="study_id",
        help="Grouping key for correlated effects; use study_id or a metadata field",
    )
    multilevel_parser.add_argument("--grid-points", type=int, default=21, help="Grid width for the variance search")
    multilevel_parser.add_argument("--refine-steps", type=int, default=3, help="Number of grid-refinement passes")

    network_parser = subparsers.add_parser(
        "network-meta",
        help="Run contrast-based network meta-analysis on a network contrast CSV",
    )
    network_parser.add_argument("--input", required=True, help="CSV file path")
    network_parser.add_argument("--reference-treatment", default="", help="Reference treatment label")
    network_parser.add_argument("--outcome", default="", help="Filter to a single outcome label")
    network_parser.add_argument(
        "--model",
        default="random",
        choices=["fixed", "random"],
        help="Network model",
    )
    network_parser.add_argument(
        "--larger-effect-is-better",
        action="store_true",
        help="Interpret larger effects as better when ranking treatments",
    )

    multivariate_vector_parser = subparsers.add_parser(
        "multivariate-meta",
        help="Run multivariate meta-analysis on a wide correlated-outcome CSV",
    )
    multivariate_vector_parser.add_argument("--input", required=True, help="CSV file path")
    multivariate_vector_parser.add_argument(
        "--model",
        default="random",
        choices=["fixed", "random"],
        help="Multivariate model",
    )
    multivariate_vector_parser.add_argument(
        "--tau-method",
        default="pm",
        choices=["dl", "pm"],
        help="Outcome-wise tau-squared estimator for random-effects pooling",
    )

    ihme_parser = subparsers.add_parser("ihme-template", help="Write the IHME registry template CSV")
    ihme_parser.add_argument(
        "--output",
        default=str(default_template_path()),
        help="Where to write the template CSV",
    )

    return parser


def _demo_meta_analysis() -> dict[str, float | str | int]:
    counts = [
        ("trial-a", 12, 88, 20, 80),
        ("trial-b", 16, 104, 28, 92),
        ("trial-c", 9, 81, 18, 72),
    ]
    effects: list[StudyEffect] = []
    for study_id, a, b, c, d in counts:
        effect_size, variance = log_odds_ratio(a, b, c, d)
        effects.append(
            StudyEffect(
                study_id=study_id,
                label=study_id,
                effect_size=effect_size,
                variance=variance,
                outcome="mortality",
                comparator="control",
                metadata={"source": "trial", "year": 2024, "sample_size": a + b + c + d},
            )
        )

    result = run_meta_analysis(effects, model="random")
    return {
        "k": result.k,
        "model": result.model,
        "pooled_log_odds_ratio": round(result.pooled_effect, 4),
        "pooled_odds_ratio": round(odds_ratio_from_log_effect(result.pooled_effect), 4),
        "ci_low": round(result.ci_low, 4),
        "ci_high": round(result.ci_high, 4),
        "i_squared": round(result.i_squared, 4),
    }


def _run_smoke_test() -> dict[str, object]:
    who_client = WHOClient()
    pubmed_client = PubMedClient()
    ctgov_client = ClinicalTrialsGovClient()
    who_url = who_client.build_indicator_url("WHOSIS_000001", top=5)
    benchmark = run_demo_benchmark()
    literature_benchmark = run_literature_screening_benchmark()
    meta_demo = _demo_meta_analysis()
    extraction_demo = extract_effect_candidates_from_text(
        "The odds ratio was 0.75 (95% CI 0.60 to 0.90).",
        source_record_id="demo-text",
        source="demo",
        outcome="demo outcome",
    )
    advanced_demo_effects = load_effects_dataset(
        Path(__file__).resolve().parents[1] / "data" / "examples" / "stroke_binary_trials.csv",
        dataset_format="binary",
        measure="or",
    )
    advanced_demo_summary = format_advanced_summary(
        summarize_effects(
            advanced_demo_effects,
            model="random",
            ci_method="hartung-knapp",
            tau_method="pm",
        ),
        effect_scale=str(advanced_demo_effects[0].metadata.get("effect_scale", "identity")),
    )
    prevalence_demo_effects = load_effects_dataset(
        Path(__file__).resolve().parents[1] / "data" / "examples" / "hypertension_prevalence_surveys.csv",
        dataset_format="proportion",
        measure="logit",
    )
    prevalence_demo_summary = format_advanced_summary(
        summarize_effects(
            prevalence_demo_effects,
            model="random",
            ci_method="hartung-knapp",
            tau_method="pm",
        ),
        effect_scale=str(prevalence_demo_effects[0].metadata.get("effect_scale", "identity")),
    )
    bayesian_demo_summary = format_advanced_summary(
        bayesian_random_effects(
            advanced_demo_effects,
            tau_prior="half-normal",
            tau_scale=0.2,
            grid_points=151,
        ),
        effect_scale=str(advanced_demo_effects[0].metadata.get("effect_scale", "identity")),
    )
    clustered_demo_effects = load_effects_dataset(
        Path(__file__).resolve().parents[1] / "data" / "examples" / "stroke_clustered_effects.csv",
        dataset_format="effects",
    )
    multilevel_demo_summary = format_advanced_summary(
        multilevel_meta_analysis(
            clustered_demo_effects,
            "study_id",
            grid_points=11,
            refine_steps=2,
        ),
        effect_scale=str(clustered_demo_effects[0].metadata.get("effect_scale", "identity")),
    )
    network_demo_contrasts = load_network_contrast_csv(
        Path(__file__).resolve().parents[1] / "data" / "examples" / "stroke_network_contrasts.csv"
    )
    network_demo_summary = _format_network_summary(
        network_meta_analysis(
            network_demo_contrasts,
            reference_treatment="Placebo",
            model="random",
        ),
        effect_scale=str(network_demo_contrasts[0].metadata.get("effect_scale", "identity")),
    )
    multivariate_demo_studies = load_multivariate_wide_csv(
        Path(__file__).resolve().parents[1] / "data" / "examples" / "stroke_multivariate_wide.csv"
    )
    multivariate_demo_summary = _format_multivariate_summary(
        multivariate_meta_analysis(multivariate_demo_studies, model="random", tau_method="pm"),
        effect_scale=str(multivariate_demo_studies[0].metadata.get("effect_scale", "identity")),
    )
    return {
        "who_url": who_url,
        "pubmed_url": pubmed_client.build_esearch_url("stroke prevention", retmax=5),
        "ctgov_url": ctgov_client.build_studies_url(condition="stroke", term="aspirin", page_size=5),
        "benchmark": benchmark,
        "literature_benchmark": literature_benchmark,
        "meta_demo": meta_demo,
        "extraction_demo": [asdict(candidate) for candidate in extraction_demo],
        "advanced_demo": advanced_demo_summary,
        "bayesian_demo": bayesian_demo_summary,
        "multilevel_demo": multilevel_demo_summary,
        "network_demo": network_demo_summary,
        "multivariate_demo": multivariate_demo_summary,
        "prevalence_demo": prevalence_demo_summary,
        "demo_effect_count": len(demo_effects()),
    }


def _format_network_summary(result: dict[str, object], *, effect_scale: str) -> dict[str, object]:
    formatted = dict(result)
    formatted["treatments"] = [
        {
            **item,
            "estimate": back_transform(float(item["effect"]), effect_scale),
            "estimate_ci_low": back_transform(float(item["ci_low"]), effect_scale),
            "estimate_ci_high": back_transform(float(item["ci_high"]), effect_scale),
        }
        for item in result["treatments"]
    ]
    formatted["pairwise_effects"] = [
        {
            **item,
            "estimate": back_transform(float(item["effect"]), effect_scale),
            "estimate_ci_low": back_transform(float(item["ci_low"]), effect_scale),
            "estimate_ci_high": back_transform(float(item["ci_high"]), effect_scale),
        }
        for item in result["pairwise_effects"]
    ]
    return formatted


def _format_multivariate_summary(result: dict[str, object], *, effect_scale: str) -> dict[str, object]:
    formatted = dict(result)
    formatted["outcomes"] = [
        {
            **item,
            "estimate": back_transform(float(item["pooled_effect"]), effect_scale),
            "estimate_ci_low": back_transform(float(item["ci_low"]), effect_scale),
            "estimate_ci_high": back_transform(float(item["ci_high"]), effect_scale),
            "estimate_prediction_low": back_transform(float(item["prediction_low"]), effect_scale),
            "estimate_prediction_high": back_transform(float(item["prediction_high"]), effect_scale),
        }
        for item in result["outcomes"]
    ]
    return formatted


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke-test":
        print(json.dumps(_run_smoke_test(), indent=2, sort_keys=True))
        return 0

    if args.command == "meta-demo":
        print(json.dumps(_demo_meta_analysis(), indent=2, sort_keys=True))
        return 0

    if args.command == "benchmark":
        print(json.dumps(run_demo_benchmark(), indent=2, sort_keys=True))
        return 0

    if args.command == "literature-benchmark":
        print(json.dumps(run_literature_screening_benchmark(), indent=2, sort_keys=True))
        return 0

    if args.command == "who-url":
        client = WHOClient()
        print(client.build_indicator_url(args.indicator, top=args.top, filter_expr=args.filter or None))
        return 0

    if args.command == "pubmed-url":
        client = PubMedClient()
        print(client.build_esearch_url(args.term, retmax=args.retmax, sort=args.sort))
        return 0

    if args.command == "pubmed-search":
        client = PubMedClient()
        records = client.search_and_summarize(args.term, retmax=args.retmax, sort=args.sort)
        print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
        return 0

    if args.command == "pubmed-extract":
        client = PubMedClient()
        articles = client.fetch_abstracts([args.pmid])
        if not articles:
            print("[]")
            return 0
        article = articles[0]
        candidates = extract_effect_candidates_from_text(
            article.abstract,
            source_record_id=f"pubmed:{article.pmid}",
            source="pubmed",
            outcome=article.title,
        )
        print(json.dumps([asdict(candidate) for candidate in candidates], indent=2, sort_keys=True))
        return 0

    if args.command == "ctgov-url":
        client = ClinicalTrialsGovClient()
        print(
            client.build_studies_url(
                term=args.term or None,
                condition=args.condition or None,
                intervention=args.intervention or None,
                page_size=args.page_size,
            )
        )
        return 0

    if args.command == "ctgov-search":
        client = ClinicalTrialsGovClient()
        records = client.search_studies(
            term=args.term or None,
            condition=args.condition or None,
            intervention=args.intervention or None,
            page_size=args.page_size,
        )
        print(json.dumps([asdict(record) for record in records], indent=2, sort_keys=True))
        return 0

    if args.command == "ctgov-extract":
        client = ClinicalTrialsGovClient()
        study_payload = client.fetch_study(args.nct_id)
        candidates = extract_ctgov_effect_candidates(study_payload)
        print(json.dumps([asdict(candidate) for candidate in candidates], indent=2, sort_keys=True))
        return 0

    if args.command == "review-pipeline":
        if not args.term and not args.pubmed_pmid and not args.ctgov_nct_id:
            parser.error("review-pipeline requires --term or at least one seeded --pubmed-pmid/--ctgov-nct-id")
        result = run_review_pipeline(
            term=args.term,
            condition=args.condition or None,
            intervention=args.intervention or None,
            pubmed_retmax=args.pubmed_retmax,
            ctgov_page_size=args.ctgov_page_size,
            include_secondary_evidence=args.include_secondary_evidence,
            min_studies=args.min_studies,
            pubmed_pmids=args.pubmed_pmid,
            ctgov_nct_ids=args.ctgov_nct_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "advanced-meta":
        effects = load_effects_dataset(args.input, dataset_format=args.format, measure=args.measure)
        if args.outcome:
            effects = [effect for effect in effects if effect.outcome == args.outcome]
        if len(effects) < 2:
            parser.error("advanced-meta requires at least two effects after filtering")

        distinct_outcomes = sorted({effect.outcome for effect in effects if effect.outcome})
        distinct_metrics = sorted({str(effect.metadata.get("metric", "")) for effect in effects if effect.metadata.get("metric", "")})
        distinct_scales = sorted({str(effect.metadata.get("effect_scale", "identity")) for effect in effects})
        if len(distinct_outcomes) > 1:
            parser.error("advanced-meta cannot pool multiple outcomes at once; filter with --outcome")
        if len(distinct_metrics) > 1 or len(distinct_scales) > 1:
            parser.error("advanced-meta cannot pool mixed effect metrics or scales in one run")

        effect_scale = str(effects[0].metadata.get("effect_scale", "identity"))
        summary = format_advanced_summary(
            summarize_effects(
                effects,
                model=args.model,
                ci_method=args.ci_method,
                tau_method=args.tau_method,
            ),
            effect_scale=effect_scale,
        )
        result: dict[str, object] = {
            "dataset": {
                "input": str(args.input),
                "format": args.format,
                "measure": args.measure,
                "outcome": distinct_outcomes[0] if distinct_outcomes else "",
                "metric": distinct_metrics[0] if distinct_metrics else "",
                "effect_scale": effect_scale,
                "k": len(effects),
            },
            "summary": summary,
            "egger_test": egger_test(effects),
        }
        if not args.skip_leave_one_out:
            result["leave_one_out"] = [
                format_advanced_summary(item, effect_scale=effect_scale)
                for item in leave_one_out_analysis(
                    effects,
                    model=args.model,
                    ci_method=args.ci_method,
                    tau_method=args.tau_method,
                )
            ]
        if args.subgroup_key:
            subgroup_result = subgroup_analysis(
                effects,
                args.subgroup_key,
                model=args.model,
                ci_method=args.ci_method,
                tau_method=args.tau_method,
            )
            subgroup_result["groups"] = [
                format_advanced_summary(group, effect_scale=effect_scale)
                for group in subgroup_result["groups"]
            ]
            result["subgroup_analysis"] = subgroup_result
        if args.cumulative_key:
            result["cumulative_analysis"] = [
                format_advanced_summary(item, effect_scale=effect_scale)
                for item in cumulative_meta_analysis(
                    effects,
                    args.cumulative_key,
                    model=args.model,
                    ci_method=args.ci_method,
                    tau_method=args.tau_method,
                )
            ]
        if args.meta_regression_key:
            covariate_keys = [
                part.strip()
                for value in args.meta_regression_key
                for part in value.split(",")
                if part.strip()
            ]
            result["meta_regression"] = meta_regression(
                effects,
                covariate_keys,
                model=args.model,
                tau_method=args.tau_method,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "bayesian-meta":
        effects = load_effects_dataset(args.input, dataset_format=args.format, measure=args.measure)
        if args.outcome:
            effects = [effect for effect in effects if effect.outcome == args.outcome]
        if len(effects) < 2:
            parser.error("bayesian-meta requires at least two effects after filtering")

        distinct_outcomes = sorted({effect.outcome for effect in effects if effect.outcome})
        distinct_metrics = sorted({str(effect.metadata.get("metric", "")) for effect in effects if effect.metadata.get("metric", "")})
        distinct_scales = sorted({str(effect.metadata.get("effect_scale", "identity")) for effect in effects})
        if len(distinct_outcomes) > 1:
            parser.error("bayesian-meta cannot pool multiple outcomes at once; filter with --outcome")
        if len(distinct_metrics) > 1 or len(distinct_scales) > 1:
            parser.error("bayesian-meta cannot pool mixed effect metrics or scales in one run")

        effect_scale = str(effects[0].metadata.get("effect_scale", "identity"))
        result = {
            "dataset": {
                "input": str(args.input),
                "format": args.format,
                "measure": args.measure,
                "outcome": distinct_outcomes[0] if distinct_outcomes else "",
                "metric": distinct_metrics[0] if distinct_metrics else "",
                "effect_scale": effect_scale,
                "k": len(effects),
            },
            "posterior": format_advanced_summary(
                bayesian_random_effects(
                    effects,
                    mu_prior_mean=args.mu_prior_mean,
                    mu_prior_sd=args.mu_prior_sd,
                    tau_prior=args.tau_prior,
                    tau_scale=args.tau_scale,
                    grid_points=args.grid_points,
                ),
                effect_scale=effect_scale,
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "multilevel-meta":
        effects = load_effects_dataset(args.input, dataset_format=args.format, measure=args.measure)
        if args.outcome:
            effects = [effect for effect in effects if effect.outcome == args.outcome]
        if len(effects) < 2:
            parser.error("multilevel-meta requires at least two effects after filtering")

        distinct_outcomes = sorted({effect.outcome for effect in effects if effect.outcome})
        distinct_metrics = sorted({str(effect.metadata.get("metric", "")) for effect in effects if effect.metadata.get("metric", "")})
        distinct_scales = sorted({str(effect.metadata.get("effect_scale", "identity")) for effect in effects})
        if len(distinct_outcomes) > 1:
            parser.error("multilevel-meta cannot pool multiple outcomes at once; filter with --outcome")
        if len(distinct_metrics) > 1 or len(distinct_scales) > 1:
            parser.error("multilevel-meta cannot pool mixed effect metrics or scales in one run")

        effect_scale = str(effects[0].metadata.get("effect_scale", "identity"))
        summary = multilevel_meta_analysis(
            effects,
            args.cluster_key,
            grid_points=args.grid_points,
            refine_steps=args.refine_steps,
        )
        result = {
            "dataset": {
                "input": str(args.input),
                "format": args.format,
                "measure": args.measure,
                "outcome": distinct_outcomes[0] if distinct_outcomes else "",
                "metric": distinct_metrics[0] if distinct_metrics else "",
                "effect_scale": effect_scale,
                "k": len(effects),
            },
            "summary": format_advanced_summary(summary, effect_scale=effect_scale),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "network-meta":
        contrasts = load_network_contrast_csv(args.input)
        if args.outcome:
            contrasts = [contrast for contrast in contrasts if contrast.outcome == args.outcome]
        if len(contrasts) < 2:
            parser.error("network-meta requires at least two contrasts after filtering")
        distinct_outcomes = sorted({contrast.outcome for contrast in contrasts if contrast.outcome})
        distinct_metrics = sorted({str(contrast.metadata.get("metric", "")) for contrast in contrasts if contrast.metadata.get("metric", "")})
        distinct_scales = sorted({str(contrast.metadata.get("effect_scale", "identity")) for contrast in contrasts})
        if len(distinct_outcomes) > 1:
            parser.error("network-meta cannot pool multiple outcomes at once; filter with --outcome")
        if len(distinct_metrics) > 1 or len(distinct_scales) > 1:
            parser.error("network-meta cannot pool mixed effect metrics or scales in one run")
        effect_scale = str(contrasts[0].metadata.get("effect_scale", "identity"))
        result = {
            "dataset": {
                "input": str(args.input),
                "outcome": distinct_outcomes[0] if distinct_outcomes else "",
                "metric": distinct_metrics[0] if distinct_metrics else "",
                "effect_scale": effect_scale,
                "k": len(contrasts),
            },
            "summary": _format_network_summary(
                network_meta_analysis(
                    contrasts,
                    reference_treatment=args.reference_treatment,
                    model=args.model,
                    smaller_effect_is_better=not args.larger_effect_is_better,
                ),
                effect_scale=effect_scale,
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "multivariate-meta":
        studies = load_multivariate_wide_csv(args.input)
        if len(studies) < 2:
            parser.error("multivariate-meta requires at least two studies")
        effect_scale = str(studies[0].metadata.get("effect_scale", "identity"))
        metrics = sorted({str(study.metadata.get("metric", "")) for study in studies if study.metadata.get("metric", "")})
        result = {
            "dataset": {
                "input": str(args.input),
                "metric": metrics[0] if metrics else "",
                "effect_scale": effect_scale,
                "k": len(studies),
                "outcomes": studies[0].outcomes,
            },
            "summary": _format_multivariate_summary(
                multivariate_meta_analysis(studies, model=args.model, tau_method=args.tau_method),
                effect_scale=effect_scale,
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "ihme-template":
        output = Path(args.output)
        write_template_catalog(output)
        print(output)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
