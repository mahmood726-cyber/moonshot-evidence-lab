from __future__ import annotations

from math import exp, log, sqrt
from typing import Iterable

from .models import MetaAnalysisResult, StudyEffect

Z_975 = 1.959963984540054
TAU_METHODS = {"dl", "pm"}


def log_odds_ratio(
    treatment_events: int,
    treatment_non_events: int,
    control_events: int,
    control_non_events: int,
    continuity_correction: float = 0.5,
) -> tuple[float, float]:
    counts = [
        float(treatment_events),
        float(treatment_non_events),
        float(control_events),
        float(control_non_events),
    ]
    if any(count < 0 for count in counts):
        raise ValueError("counts must be non-negative")

    if any(count == 0 for count in counts):
        counts = [count + continuity_correction for count in counts]

    a, b, c, d = counts
    effect = log((a * d) / (b * c))
    variance = (1.0 / a) + (1.0 / b) + (1.0 / c) + (1.0 / d)
    return effect, variance


def odds_ratio_from_log_effect(effect_size: float) -> float:
    return exp(effect_size)


def _weights(effects: Iterable[StudyEffect], tau_squared: float = 0.0) -> list[float]:
    return [1.0 / (effect.variance + tau_squared) for effect in effects]


def _weighted_mean(effects: Iterable[StudyEffect], weights: Iterable[float]) -> float:
    numerator = sum(weight * effect.effect_size for weight, effect in zip(weights, effects))
    denominator = sum(weights)
    return numerator / denominator


def _cochran_q(effects: list[StudyEffect], weights: list[float]) -> tuple[float, float]:
    pooled_effect = _weighted_mean(effects, weights)
    q = sum(
        weight * (effect.effect_size - pooled_effect) ** 2
        for weight, effect in zip(weights, effects)
    )
    return pooled_effect, q


def _der_simonian_laird_tau_squared(effects: list[StudyEffect]) -> float:
    fixed_weights = _weights(effects)
    fixed_weight_total = sum(fixed_weights)
    _, q = _cochran_q(effects, fixed_weights)
    c = fixed_weight_total - (sum(weight ** 2 for weight in fixed_weights) / fixed_weight_total)
    if c <= 0:
        return 0.0
    return max(0.0, (q - (len(effects) - 1)) / c)


def _generalized_q(effects: list[StudyEffect], tau_squared: float) -> float:
    weights = _weights(effects, tau_squared)
    pooled_effect = _weighted_mean(effects, weights)
    return sum(
        weight * (effect.effect_size - pooled_effect) ** 2
        for weight, effect in zip(weights, effects)
    )


def _paule_mandel_tau_squared(
    effects: list[StudyEffect],
    *,
    tolerance: float = 1.0e-10,
    max_iter: int = 200,
) -> float:
    target = float(len(effects) - 1)
    q_at_zero = _generalized_q(effects, 0.0)
    if q_at_zero <= target:
        return 0.0

    lower = 0.0
    upper = 1.0
    q_upper = _generalized_q(effects, upper)
    while q_upper > target and upper < 1.0e6:
        lower = upper
        upper *= 2.0
        q_upper = _generalized_q(effects, upper)

    for _ in range(max_iter):
        midpoint = (lower + upper) / 2.0
        q_mid = _generalized_q(effects, midpoint)
        if abs(q_mid - target) <= tolerance:
            return midpoint
        if q_mid > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def estimate_tau_squared(
    effects: list[StudyEffect],
    *,
    tau_method: str = "dl",
) -> float:
    tau_method_name = tau_method.strip().lower()
    if tau_method_name not in TAU_METHODS:
        raise ValueError("tau_method must be 'dl' or 'pm'")
    if tau_method_name == "dl":
        return _der_simonian_laird_tau_squared(effects)
    return _paule_mandel_tau_squared(effects)


def run_meta_analysis(
    effects: list[StudyEffect],
    model: str = "random",
    *,
    tau_method: str = "dl",
) -> MetaAnalysisResult:
    if len(effects) < 2:
        raise ValueError("at least two effects are required for meta-analysis")

    model_name = model.strip().lower()
    if model_name not in {"fixed", "random"}:
        raise ValueError("model must be 'fixed' or 'random'")

    fixed_weights = _weights(effects)
    _, q = _cochran_q(effects, fixed_weights)
    degrees_of_freedom = len(effects) - 1
    tau_squared = 0.0
    tau_method_name = "fixed"
    if model_name == "random":
        tau_method_name = tau_method.strip().lower()
        tau_squared = estimate_tau_squared(effects, tau_method=tau_method_name)

    weights = fixed_weights if model_name == "fixed" else _weights(effects, tau_squared)
    weight_total = sum(weights)
    pooled_effect = _weighted_mean(effects, weights)
    standard_error = sqrt(1.0 / weight_total)
    ci_low = pooled_effect - (Z_975 * standard_error)
    ci_high = pooled_effect + (Z_975 * standard_error)
    i_squared = 0.0 if q <= 0 else max(0.0, ((q - degrees_of_freedom) / q) * 100.0)
    h_squared = 1.0 if q <= 0 else max(1.0, q / max(float(degrees_of_freedom), 1.0))
    h = sqrt(h_squared)

    return MetaAnalysisResult(
        model=model_name,
        pooled_effect=pooled_effect,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        tau_squared=tau_squared,
        tau_method=tau_method_name,
        q=q,
        i_squared=i_squared,
        h=h,
        h_squared=h_squared,
        k=len(effects),
    )
