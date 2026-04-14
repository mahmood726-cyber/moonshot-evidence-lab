from __future__ import annotations

import re
from collections import Counter, defaultdict
from math import exp
from typing import Any

from .connectors.clinicaltrials import ClinicalTrialRecord, ClinicalTrialsGovClient
from .connectors.pubmed import PubMedArticle, PubMedClient
from .extraction import extract_ctgov_effect_candidates, extract_effect_candidates_from_text
from .literature import clinical_trial_to_record, pubmed_article_to_record
from .meta_analysis import run_meta_analysis
from .models import ExtractedEffectCandidate, StudyEffect
from .program_surface import DEFAULT_POLICY, choose_meta_model, include_record

SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    value = SLUG_PATTERN.sub("_", text.strip().lower()).strip("_")
    return value or "unspecified_outcome"


def canonical_outcome_tag(*texts: str) -> str:
    haystack = " ".join(texts).lower()
    if "hemorrhagic stroke" in haystack:
        return "hemorrhagic_stroke"
    if "stroke" in haystack:
        return "stroke"
    if "cardiovascular mortality" in haystack:
        return "cardiovascular_mortality"
    if "all-cause mortality" in haystack or "all cause mortality" in haystack:
        return "all_cause_mortality"
    if "mortality" in haystack:
        return "mortality"
    if "myocardial infarction" in haystack:
        return "myocardial_infarction"
    if "gastrointestinal bleeding" in haystack:
        return "gastrointestinal_bleeding"
    if "major bleeding" in haystack:
        return "major_bleeding"
    if "bleeding" in haystack:
        return "bleeding"
    if "pulsatility index" in haystack or "pi " in haystack or " pi" in haystack:
        return "pulsatility_index"
    first_non_empty = next((text for text in texts if text.strip()), "")
    return _slugify(first_non_empty[:80])


def _is_secondary_pubmed(article: PubMedArticle) -> bool:
    publication_types = [str(item).strip().lower() for item in article.metadata.get("pubtype", [])]
    return any(
        token in publication_type
        for publication_type in publication_types
        for token in ("review", "meta-analysis")
    )


def _deduplicate_candidates(
    candidates: list[ExtractedEffectCandidate],
) -> list[ExtractedEffectCandidate]:
    best_by_key: dict[tuple[str, str, str], ExtractedEffectCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.source_record_id,
            str(candidate.metadata.get("outcome_tag", "")),
            candidate.metric,
        )
        current = best_by_key.get(key)
        if current is None or candidate.variance < current.variance:
            best_by_key[key] = candidate
    return list(best_by_key.values())


def _candidate_label(candidate: ExtractedEffectCandidate) -> str:
    outcome_tag = str(candidate.metadata.get("outcome_tag", "")).strip()
    if outcome_tag:
        return f"{candidate.source_record_id}:{outcome_tag}"
    return candidate.source_record_id


def _candidate_to_study_effect(
    candidate: ExtractedEffectCandidate,
    *,
    ordinal: int,
) -> StudyEffect:
    return candidate.to_study_effect(
        study_id=f"{candidate.source_record_id}:{ordinal}",
        label=_candidate_label(candidate),
    )


def _natural_scale_summary(effect_scale: str, pooled_effect: float, ci_low: float, ci_high: float) -> tuple[float, float, float]:
    if effect_scale == "log":
        return exp(pooled_effect), exp(ci_low), exp(ci_high)
    return pooled_effect, ci_low, ci_high


def _prepare_candidate_metadata(
    candidate: ExtractedEffectCandidate,
    *,
    record_year: int | None,
    source_title: str,
    secondary_evidence: bool,
) -> ExtractedEffectCandidate:
    outcome_tag = canonical_outcome_tag(candidate.outcome, candidate.snippet, source_title)
    candidate.metadata = {
        **candidate.metadata,
        "year": record_year,
        "source_title": source_title,
        "secondary_evidence": secondary_evidence,
        "outcome_tag": outcome_tag,
    }
    return candidate


def collect_pubmed_candidates(
    term: str,
    *,
    retmax: int,
    client: PubMedClient,
    policy=DEFAULT_POLICY,
    include_secondary_evidence: bool = False,
    pmids: list[str] | None = None,
) -> tuple[list[PubMedArticle], list[ExtractedEffectCandidate], dict[str, int]]:
    articles = client.fetch_articles(pmids) if pmids else client.search_and_fetch_abstracts(term, retmax=retmax)
    candidates: list[ExtractedEffectCandidate] = []
    stats = {"screened_in": 0, "screened_out": 0, "secondary_excluded": 0}

    for article in articles:
        record = pubmed_article_to_record(article)
        if not include_record(record.metadata, policy=policy):
            stats["screened_out"] += 1
            continue

        secondary_evidence = _is_secondary_pubmed(article)
        if secondary_evidence and not include_secondary_evidence:
            stats["secondary_excluded"] += 1
            continue

        stats["screened_in"] += 1
        extracted = extract_effect_candidates_from_text(
            article.abstract,
            source_record_id=record.record_id,
            source="pubmed",
            outcome=record.title,
        )
        for candidate in extracted:
            candidates.append(
                _prepare_candidate_metadata(
                    candidate,
                    record_year=record.year,
                    source_title=record.title,
                    secondary_evidence=secondary_evidence,
                )
            )

    return articles, candidates, stats


def collect_ctgov_candidates(
    *,
    term: str | None,
    condition: str | None,
    intervention: str | None,
    page_size: int,
    client: ClinicalTrialsGovClient,
    policy=DEFAULT_POLICY,
    nct_ids: list[str] | None = None,
) -> tuple[list[ClinicalTrialRecord], list[ExtractedEffectCandidate], dict[str, int]]:
    if nct_ids:
        records = []
        for nct_id in nct_ids:
            payload = client.fetch_study(nct_id)
            records.extend(client.normalize_studies(payload))
    else:
        records = client.search_studies(
            term=term,
            condition=condition,
            intervention=intervention,
            page_size=page_size,
        )
    candidates: list[ExtractedEffectCandidate] = []
    stats = {"screened_in": 0, "screened_out": 0, "fetch_failures": 0}

    for trial in records:
        evidence_record = clinical_trial_to_record(trial)
        if not include_record(evidence_record.metadata, policy=policy):
            stats["screened_out"] += 1
            continue

        try:
            study_payload = client.fetch_study(trial.nct_id)
        except Exception:
            stats["fetch_failures"] += 1
            continue

        stats["screened_in"] += 1
        extracted = extract_ctgov_effect_candidates(study_payload)
        for candidate in extracted:
            candidates.append(
                _prepare_candidate_metadata(
                    candidate,
                    record_year=evidence_record.year,
                    source_title=evidence_record.title,
                    secondary_evidence=False,
                )
            )

    return records, candidates, stats


def pool_effect_candidates(
    candidates: list[ExtractedEffectCandidate],
    *,
    min_studies: int = 2,
    policy=DEFAULT_POLICY,
) -> tuple[list[StudyEffect], list[dict[str, Any]], list[dict[str, Any]]]:
    deduplicated = _deduplicate_candidates(candidates)
    effects = [
        _candidate_to_study_effect(candidate, ordinal=index + 1)
        for index, candidate in enumerate(deduplicated)
    ]

    grouped: dict[tuple[str, str, str], list[StudyEffect]] = defaultdict(list)
    for effect in effects:
        key = (
            str(effect.metadata.get("outcome_tag", "")),
            str(effect.metadata.get("metric", "")),
            str(effect.metadata.get("effect_scale", "")),
        )
        grouped[key].append(effect)

    candidate_groups: list[dict[str, Any]] = []
    pooled: list[dict[str, Any]] = []

    for key, grouped_effects in grouped.items():
        outcome_tag, metric, effect_scale = key
        candidate_groups.append(
            {
                "outcome_tag": outcome_tag,
                "metric": metric,
                "effect_scale": effect_scale,
                "k": len(grouped_effects),
                "study_ids": [effect.study_id for effect in grouped_effects],
            }
        )
        if len(grouped_effects) < min_studies:
            continue

        model = choose_meta_model(len(grouped_effects), policy=policy)
        result = run_meta_analysis(grouped_effects, model=model)
        natural_estimate, natural_low, natural_high = _natural_scale_summary(
            effect_scale,
            result.pooled_effect,
            result.ci_low,
            result.ci_high,
        )

        pooled.append(
            {
                "outcome_tag": outcome_tag,
                "metric": metric,
                "effect_scale": effect_scale,
                "k": result.k,
                "model": result.model,
                "pooled_effect": round(result.pooled_effect, 6),
                "pooled_estimate": round(natural_estimate, 6),
                "ci_low": round(result.ci_low, 6),
                "ci_high": round(result.ci_high, 6),
                "estimate_ci_low": round(natural_low, 6),
                "estimate_ci_high": round(natural_high, 6),
                "i_squared": round(result.i_squared, 4),
                "tau_squared": round(result.tau_squared, 6),
                "study_ids": [effect.study_id for effect in grouped_effects],
                "source_breakdown": dict(Counter(str(effect.metadata.get("source", "")) for effect in grouped_effects)),
            }
        )

    candidate_groups.sort(key=lambda item: (-int(item["k"]), str(item["outcome_tag"]), str(item["metric"])))
    pooled.sort(key=lambda item: (-int(item["k"]), str(item["outcome_tag"]), str(item["metric"])))
    return effects, candidate_groups, pooled


def run_review_pipeline(
    *,
    term: str,
    condition: str | None = None,
    intervention: str | None = None,
    pubmed_retmax: int = 5,
    ctgov_page_size: int = 5,
    include_secondary_evidence: bool = False,
    min_studies: int = 2,
    pubmed_pmids: list[str] | None = None,
    ctgov_nct_ids: list[str] | None = None,
    pubmed_client: PubMedClient | None = None,
    ctgov_client: ClinicalTrialsGovClient | None = None,
    policy=DEFAULT_POLICY,
) -> dict[str, Any]:
    pubmed = pubmed_client or PubMedClient()
    ctgov = ctgov_client or ClinicalTrialsGovClient()
    errors: list[dict[str, str]] = []

    try:
        pubmed_articles, pubmed_candidates, pubmed_stats = collect_pubmed_candidates(
            term,
            retmax=pubmed_retmax,
            client=pubmed,
            policy=policy,
            include_secondary_evidence=include_secondary_evidence,
            pmids=pubmed_pmids,
        )
    except Exception as exc:
        pubmed_articles, pubmed_candidates = [], []
        pubmed_stats = {"screened_in": 0, "screened_out": 0, "secondary_excluded": 0}
        errors.append({"source": "pubmed", "message": str(exc)})

    try:
        ctgov_records, ctgov_candidates, ctgov_stats = collect_ctgov_candidates(
            term=term or None,
            condition=condition,
            intervention=intervention,
            page_size=ctgov_page_size,
            client=ctgov,
            policy=policy,
            nct_ids=ctgov_nct_ids,
        )
    except Exception as exc:
        ctgov_records, ctgov_candidates = [], []
        ctgov_stats = {"screened_in": 0, "screened_out": 0, "fetch_failures": 0}
        errors.append({"source": "ctgov", "message": str(exc)})

    all_candidates = [*pubmed_candidates, *ctgov_candidates]
    effects, candidate_groups, pooled = pool_effect_candidates(
        all_candidates,
        min_studies=min_studies,
        policy=policy,
    )

    return {
        "query": {
            "term": term,
            "condition": condition or "",
            "intervention": intervention or "",
            "pubmed_retmax": pubmed_retmax,
            "ctgov_page_size": ctgov_page_size,
            "include_secondary_evidence": include_secondary_evidence,
            "min_studies": min_studies,
            "pubmed_pmids": pubmed_pmids or [],
            "ctgov_nct_ids": ctgov_nct_ids or [],
        },
        "records": {
            "pubmed_found": len(pubmed_articles),
            "pubmed_screened_in": pubmed_stats["screened_in"],
            "pubmed_screened_out": pubmed_stats["screened_out"],
            "pubmed_secondary_excluded": pubmed_stats["secondary_excluded"],
            "ctgov_found": len(ctgov_records),
            "ctgov_screened_in": ctgov_stats["screened_in"],
            "ctgov_screened_out": ctgov_stats["screened_out"],
            "ctgov_fetch_failures": ctgov_stats["fetch_failures"],
        },
        "candidates": {
            "pubmed": len(pubmed_candidates),
            "ctgov": len(ctgov_candidates),
            "total": len(all_candidates),
            "deduplicated_effects": len(effects),
        },
        "errors": errors,
        "candidate_groups": candidate_groups,
        "pooled_analyses": pooled,
    }
