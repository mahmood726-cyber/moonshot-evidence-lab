from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StudyEffect:
    study_id: str
    label: str
    effect_size: float
    variance: float
    outcome: str = ""
    comparator: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variance <= 0:
            raise ValueError("variance must be positive")


@dataclass(slots=True)
class EvidenceRecord:
    record_id: str
    source: str
    title: str
    year: int | None = None
    abstract: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedEffectCandidate:
    source_record_id: str
    source: str
    metric: str
    effect_size: float
    variance: float
    effect_scale: str
    outcome: str = ""
    comparator: str = ""
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    snippet: str = ""
    extraction_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_study_effect(
        self,
        *,
        study_id: str | None = None,
        label: str | None = None,
    ) -> StudyEffect:
        return StudyEffect(
            study_id=study_id or self.source_record_id,
            label=label or self.source_record_id,
            effect_size=self.effect_size,
            variance=self.variance,
            outcome=self.outcome,
            comparator=self.comparator,
            metadata={
                "source": self.source,
                "metric": self.metric,
                "effect_scale": self.effect_scale,
                "estimate": self.estimate,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
                "snippet": self.snippet,
                "extraction_method": self.extraction_method,
                **self.metadata,
            },
        )


@dataclass(slots=True)
class MetaAnalysisResult:
    model: str
    pooled_effect: float
    standard_error: float
    ci_low: float
    ci_high: float
    tau_squared: float
    tau_method: str
    q: float
    i_squared: float
    h: float
    h_squared: float
    k: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "pooled_effect": self.pooled_effect,
            "standard_error": self.standard_error,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "tau_squared": self.tau_squared,
            "tau_method": self.tau_method,
            "q": self.q,
            "i_squared": self.i_squared,
            "h": self.h,
            "h_squared": self.h_squared,
            "k": self.k,
        }


@dataclass(slots=True)
class NetworkContrast:
    study_id: str
    label: str
    treatment_a: str
    treatment_b: str
    effect_size: float
    variance: float
    outcome: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variance <= 0:
            raise ValueError("variance must be positive")
        if not self.treatment_a or not self.treatment_b:
            raise ValueError("network contrasts require two treatment labels")


@dataclass(slots=True)
class MultivariateStudy:
    study_id: str
    label: str
    outcomes: list[str]
    effects: dict[str, float]
    covariance: list[list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        outcome_count = len(self.outcomes)
        if outcome_count == 0:
            raise ValueError("multivariate studies require at least one outcome")
        if set(self.effects) != set(self.outcomes):
            raise ValueError("effects keys must match outcomes")
        if len(self.covariance) != outcome_count:
            raise ValueError("covariance row count must match outcomes")
        for row in self.covariance:
            if len(row) != outcome_count:
                raise ValueError("covariance matrix must be square")
        for index, outcome in enumerate(self.outcomes):
            if self.covariance[index][index] <= 0:
                raise ValueError(f"variance for outcome '{outcome}' must be positive")
