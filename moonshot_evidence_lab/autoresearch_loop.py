from __future__ import annotations

from statistics import fmean

from .meta_analysis import run_meta_analysis
from .models import StudyEffect
from .program_surface import DEFAULT_POLICY, choose_meta_model, include_record, record_trust_score


def demo_effects() -> list[StudyEffect]:
    return [
        StudyEffect(
            study_id="trial-001",
            label="Community screening trial A",
            effect_size=-0.18,
            variance=0.04,
            outcome="all_cause_mortality",
            metadata={"source": "trial", "year": 2021, "sample_size": 420, "missingness": 0.05},
        ),
        StudyEffect(
            study_id="trial-002",
            label="Community screening preprint",
            effect_size=-0.10,
            variance=0.03,
            outcome="all_cause_mortality",
            metadata={"source": "preprint", "year": 2024, "sample_size": 110, "missingness": 0.02},
        ),
        StudyEffect(
            study_id="who-001",
            label="WHO indicator benchmark",
            effect_size=-0.14,
            variance=0.02,
            outcome="risk_factor_burden",
            metadata={"source": "who", "year": 2023, "sample_size": 1200, "missingness": 0.00},
        ),
        StudyEffect(
            study_id="ihme-001",
            label="IHME modeled estimate",
            effect_size=-0.12,
            variance=0.05,
            outcome="population_burden",
            metadata={"source": "ihme", "year": 2022, "sample_size": 800, "missingness": 0.10},
        ),
        StudyEffect(
            study_id="legacy-001",
            label="Old cohort estimate",
            effect_size=-0.08,
            variance=0.04,
            outcome="all_cause_mortality",
            metadata={"source": "cohort", "year": 1998, "sample_size": 220, "missingness": 0.05},
        ),
    ]


def run_demo_benchmark() -> dict[str, float | int | str]:
    effects = demo_effects()
    selected = [effect for effect in effects if include_record(effect.metadata, policy=DEFAULT_POLICY)]
    if len(selected) < 2:
        raise ValueError("benchmark selected fewer than two studies")

    trust = fmean(record_trust_score(effect.metadata) for effect in selected)
    model = choose_meta_model(len(selected), policy=DEFAULT_POLICY)
    meta = run_meta_analysis(selected, model=model)

    coverage_score = len(selected) / len(effects)
    stability_score = 1.0 - min(meta.i_squared, 100.0) / 100.0
    total_score = (0.45 * coverage_score) + (0.25 * trust) + (0.30 * stability_score)

    return {
        "selected_studies": len(selected),
        "rejected_studies": len(effects) - len(selected),
        "coverage_score": round(coverage_score, 4),
        "mean_trust_score": round(trust, 4),
        "stability_score": round(stability_score, 4),
        "total_score": round(total_score, 4),
        "model": meta.model,
        "pooled_effect": round(meta.pooled_effect, 4),
        "ci_low": round(meta.ci_low, 4),
        "ci_high": round(meta.ci_high, 4),
        "i_squared": round(meta.i_squared, 4),
    }
