from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

TRUST_PRIORS = {
    "who": 1.0,
    "ihme": 0.95,
    "systematic_review": 0.9,
    "trial": 0.9,
    "pubmed": 0.78,
    "ctgov": 0.72,
    "cohort": 0.75,
    "case_control": 0.7,
    "registry": 0.68,
    "preprint": 0.4,
}


@dataclass(slots=True, frozen=True)
class ProgramPolicy:
    min_year: int = 2000
    min_sample_size: int = 50
    max_missingness: float = 0.30
    allow_preprints: bool = False
    prefer_random_effects_from_k: int = 3
    fixed_effect_i2_threshold: float = 25.0
    minimum_trust_score: float = 0.55


DEFAULT_POLICY = ProgramPolicy()


def record_trust_score(metadata: Mapping[str, Any]) -> float:
    source = str(metadata.get("source", "")).strip().lower()
    if not source:
        return 0.5
    return TRUST_PRIORS.get(source, 0.55)


def include_record(
    metadata: Mapping[str, Any],
    policy: ProgramPolicy = DEFAULT_POLICY,
) -> bool:
    year = int(metadata.get("year", 0) or 0)
    sample_size = int(metadata.get("sample_size", 0) or 0)
    missingness = float(metadata.get("missingness", 0.0) or 0.0)
    source = str(metadata.get("source", "")).strip().lower()

    if year and year < policy.min_year:
        return False
    if sample_size and sample_size < policy.min_sample_size:
        return False
    if missingness > policy.max_missingness:
        return False
    if source == "preprint" and not policy.allow_preprints:
        return False
    if source == "ctgov" and not bool(metadata.get("has_results", False)):
        return False
    if bool(metadata.get("retracted", False)):
        return False
    if str(metadata.get("study_design", "")).strip().lower() == "protocol":
        return False

    return record_trust_score(metadata) >= policy.minimum_trust_score


def choose_meta_model(
    study_count: int,
    heterogeneity_hint: float | None = None,
    policy: ProgramPolicy = DEFAULT_POLICY,
) -> str:
    if study_count < policy.prefer_random_effects_from_k:
        return "fixed"
    if heterogeneity_hint is not None and heterogeneity_hint < policy.fixed_effect_i2_threshold:
        return "fixed"
    return "random"
