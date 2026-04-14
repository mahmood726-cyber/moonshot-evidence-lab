from __future__ import annotations

from math import asin, exp, lgamma, log, pi, sin, sqrt
from statistics import NormalDist
from typing import Any

from .meta_analysis import Z_975, run_meta_analysis
from .models import StudyEffect

NORMAL = NormalDist()
GAMMA_EPS = 3.0e-14
GAMMA_ITMAX = 200
GAMMA_FPMIN = 1.0e-300
SINGULARITY_EPS = 1.0e-12
LOG_2PI = log(2.0 * pi)


def log_risk_ratio(
    treatment_events: int,
    treatment_total: int,
    control_events: int,
    control_total: int,
    continuity_correction: float = 0.5,
) -> tuple[float, float]:
    treatment_non_events = treatment_total - treatment_events
    control_non_events = control_total - control_events
    counts = [
        float(treatment_events),
        float(treatment_non_events),
        float(control_events),
        float(control_non_events),
    ]
    if any(count < 0 for count in counts):
        raise ValueError("counts must be non-negative")
    if treatment_total <= 0 or control_total <= 0:
        raise ValueError("totals must be positive")

    if any(count == 0 for count in counts):
        counts = [count + continuity_correction for count in counts]

    a, b, c, d = counts
    effect = log((a / (a + b)) / (c / (c + d)))
    variance = (1.0 / a) - (1.0 / (a + b)) + (1.0 / c) - (1.0 / (c + d))
    return effect, variance


def risk_difference(
    treatment_events: int,
    treatment_total: int,
    control_events: int,
    control_total: int,
) -> tuple[float, float]:
    if treatment_total <= 0 or control_total <= 0:
        raise ValueError("totals must be positive")
    treatment_risk = treatment_events / treatment_total
    control_risk = control_events / control_total
    effect = treatment_risk - control_risk
    variance = (
        (treatment_risk * (1.0 - treatment_risk) / treatment_total)
        + (control_risk * (1.0 - control_risk) / control_total)
    )
    return effect, variance


def mean_difference(
    treatment_mean: float,
    treatment_sd: float,
    treatment_n: int,
    control_mean: float,
    control_sd: float,
    control_n: int,
) -> tuple[float, float]:
    if treatment_n <= 1 or control_n <= 1:
        raise ValueError("group sample sizes must exceed one")
    effect = treatment_mean - control_mean
    variance = (treatment_sd ** 2 / treatment_n) + (control_sd ** 2 / control_n)
    return effect, variance


def standardized_mean_difference(
    treatment_mean: float,
    treatment_sd: float,
    treatment_n: int,
    control_mean: float,
    control_sd: float,
    control_n: int,
) -> tuple[float, float]:
    if treatment_n <= 1 or control_n <= 1:
        raise ValueError("group sample sizes must exceed one")

    df = treatment_n + control_n - 2
    if df <= 0:
        raise ValueError("pooled standard deviation requires at least two total degrees of freedom")

    pooled_variance = (
        ((treatment_n - 1) * (treatment_sd ** 2))
        + ((control_n - 1) * (control_sd ** 2))
    ) / df
    if pooled_variance <= 0:
        raise ValueError("pooled variance must be positive")

    pooled_sd = sqrt(pooled_variance)
    cohen_d = (treatment_mean - control_mean) / pooled_sd
    correction = 1.0 - (3.0 / ((4.0 * df) - 1.0))
    hedges_g = correction * cohen_d
    variance = ((treatment_n + control_n) / (treatment_n * control_n)) + (
        (hedges_g ** 2) / (2.0 * (treatment_n + control_n - 3.94))
    )
    return hedges_g, variance


def log_ratio_of_means(
    treatment_mean: float,
    treatment_sd: float,
    treatment_n: int,
    control_mean: float,
    control_sd: float,
    control_n: int,
) -> tuple[float, float]:
    if treatment_mean <= 0 or control_mean <= 0:
        raise ValueError("means must be positive for ratio-of-means analyses")
    if treatment_n <= 1 or control_n <= 1:
        raise ValueError("group sample sizes must exceed one")

    effect = log(treatment_mean / control_mean)
    variance = (
        (treatment_sd ** 2 / (treatment_n * treatment_mean ** 2))
        + (control_sd ** 2 / (control_n * control_mean ** 2))
    )
    return effect, variance


def logit_proportion(
    events: int,
    total: int,
    continuity_correction: float = 0.5,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if events < 0 or events > total:
        raise ValueError("events must be between zero and total")

    successes = float(events)
    failures = float(total - events)
    if successes == 0.0 or failures == 0.0:
        successes += continuity_correction
        failures += continuity_correction
    effect = log(successes / failures)
    variance = (1.0 / successes) + (1.0 / failures)
    return effect, variance


def raw_proportion(
    events: int,
    total: int,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if events < 0 or events > total:
        raise ValueError("events must be between zero and total")
    proportion = events / total
    if proportion in {0.0, 1.0}:
        adjusted = (events + 0.5) / (total + 1.0)
        variance = adjusted * (1.0 - adjusted) / (total + 1.0)
        return adjusted, variance
    variance = proportion * (1.0 - proportion) / total
    return proportion, variance


def arcsine_proportion(
    events: int,
    total: int,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    if events < 0 or events > total:
        raise ValueError("events must be between zero and total")
    adjusted = (events + 0.5) / (total + 1.0)
    effect = asin(sqrt(adjusted))
    variance = 1.0 / (4.0 * (total + 0.5))
    return effect, variance


def approx_t_critical(df: int, alpha: float = 0.05) -> float:
    if df <= 0:
        return Z_975
    z = NORMAL.inv_cdf(1.0 - (alpha / 2.0))
    df_float = float(df)
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2
    return (
        z
        + ((z3 + z) / (4.0 * df_float))
        + ((5.0 * z5) + (16.0 * z3) + (3.0 * z)) / (96.0 * df_float ** 2)
        + ((3.0 * z7) + (19.0 * z5) + (17.0 * z3) - (15.0 * z)) / (384.0 * df_float ** 3)
    )


def normal_two_sided_p(statistic: float) -> float:
    return 2.0 * (1.0 - NORMAL.cdf(abs(statistic)))


def _normal_cdf(value: float, mean: float, variance: float) -> float:
    standard_deviation = sqrt(max(variance, SINGULARITY_EPS))
    return NORMAL.cdf((value - mean) / standard_deviation)


def _logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    max_value = max(values)
    return max_value + log(sum(exp(value - max_value) for value in values))


def _weighted_grid_quantile(points: list[float], weights: list[float], probability: float) -> float:
    if not points or len(points) != len(weights):
        raise ValueError("points and weights must be non-empty and aligned")
    if probability <= 0.0:
        return points[0]
    if probability >= 1.0:
        return points[-1]
    cumulative = 0.0
    previous_point = points[0]
    previous_cumulative = 0.0
    for point, weight in zip(points, weights):
        cumulative += weight
        if cumulative >= probability:
            if cumulative == previous_cumulative:
                return point
            fraction = (probability - previous_cumulative) / (cumulative - previous_cumulative)
            return previous_point + (fraction * (point - previous_point))
        previous_point = point
        previous_cumulative = cumulative
    return points[-1]


def _normal_mixture_quantile(
    means: list[float],
    variances: list[float],
    weights: list[float],
    probability: float,
    *,
    grid_points: int = 801,
) -> float:
    lower = min(mean - (7.0 * sqrt(max(variance, SINGULARITY_EPS))) for mean, variance in zip(means, variances))
    upper = max(mean + (7.0 * sqrt(max(variance, SINGULARITY_EPS))) for mean, variance in zip(means, variances))
    if upper <= lower:
        return means[0]
    point_count = max(grid_points, 101)
    step = (upper - lower) / (point_count - 1)
    grid = [lower + (index * step) for index in range(point_count)]
    mixture_cdf = [
        sum(
            weight * _normal_cdf(point, mean, variance)
            for mean, variance, weight in zip(means, variances, weights)
        )
        for point in grid
    ]
    for index, cdf_value in enumerate(mixture_cdf):
        if cdf_value >= probability:
            if index == 0:
                return grid[0]
            lower_point = grid[index - 1]
            upper_point = grid[index]
            lower_cdf = mixture_cdf[index - 1]
            if cdf_value == lower_cdf:
                return upper_point
            fraction = (probability - lower_cdf) / (cdf_value - lower_cdf)
            return lower_point + (fraction * (upper_point - lower_point))
    return grid[-1]


def _gamma_series(a: float, x: float) -> float:
    ap = a
    delta = 1.0 / a
    total = delta
    for _ in range(GAMMA_ITMAX):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * GAMMA_EPS:
            break
    return total * exp((-x) + (a * log(x)) - lgamma(a))


def _gamma_continued_fraction(a: float, x: float) -> float:
    b = x + 1.0 - a
    c = 1.0 / GAMMA_FPMIN
    d = 1.0 / max(b, GAMMA_FPMIN)
    h = d
    for i in range(1, GAMMA_ITMAX + 1):
        an = -float(i) * (float(i) - a)
        b += 2.0
        d = (an * d) + b
        if abs(d) < GAMMA_FPMIN:
            d = GAMMA_FPMIN
        c = b + (an / c)
        if abs(c) < GAMMA_FPMIN:
            c = GAMMA_FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < GAMMA_EPS:
            break
    return exp((-x) + (a * log(x)) - lgamma(a)) * h


def regularized_gamma_q(a: float, x: float) -> float:
    if a <= 0:
        raise ValueError("shape parameter must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gamma_series(a, x))
    return min(1.0, _gamma_continued_fraction(a, x))


def chi_square_sf(statistic: float, df: int) -> float:
    if df <= 0:
        return 1.0
    if statistic < 0:
        return 1.0
    return regularized_gamma_q(df / 2.0, statistic / 2.0)


def _arcsine_back_transform(effect: float) -> float:
    return max(0.0, min(1.0, sin(effect) ** 2))


def back_transform(effect: float, effect_scale: str) -> float:
    scale_name = effect_scale.strip().lower()
    if scale_name == "log":
        return exp(effect)
    if scale_name == "logit":
        return 1.0 / (1.0 + exp(-effect))
    if scale_name == "arcsine":
        return _arcsine_back_transform(effect)
    return effect


def _single_study_summary(effect: StudyEffect, *, ci_method: str) -> dict[str, float | int | str | None]:
    standard_error = sqrt(effect.variance)
    return {
        "model": "single-study",
        "ci_method": ci_method,
        "k": 1,
        "pooled_effect": effect.effect_size,
        "standard_error": standard_error,
        "ci_standard_error": standard_error,
        "ci_low": effect.effect_size - (Z_975 * standard_error),
        "ci_high": effect.effect_size + (Z_975 * standard_error),
        "tau_squared": 0.0,
        "tau_method": "fixed",
        "q": 0.0,
        "q_p_value": 1.0,
        "i_squared": 0.0,
        "h": 1.0,
        "h_squared": 1.0,
        "prediction_low": None,
        "prediction_high": None,
    }


def summarize_effects(
    effects: list[StudyEffect],
    *,
    model: str = "random",
    ci_method: str = "normal",
    include_prediction_interval: bool = True,
    tau_method: str = "dl",
) -> dict[str, float | int | str | None]:
    base_result = run_meta_analysis(effects, model=model, tau_method=tau_method)
    ci_method_name = ci_method.strip().lower()
    if ci_method_name not in {"normal", "hartung-knapp"}:
        raise ValueError("ci_method must be 'normal' or 'hartung-knapp'")

    weights = [1.0 / (effect.variance + base_result.tau_squared) for effect in effects]
    conventional_se = sqrt(1.0 / sum(weights))
    ci_critical = Z_975
    ci_se = conventional_se
    if ci_method_name == "hartung-knapp" and model.strip().lower() == "random" and len(effects) > 2:
        hk_variance = sum(
            weight * (effect.effect_size - base_result.pooled_effect) ** 2
            for weight, effect in zip(weights, effects)
        ) / ((len(effects) - 1) * sum(weights))
        ci_se = sqrt(max(hk_variance, 0.0))
        ci_critical = approx_t_critical(len(effects) - 1)

    ci_low = base_result.pooled_effect - (ci_critical * ci_se)
    ci_high = base_result.pooled_effect + (ci_critical * ci_se)

    prediction_low = None
    prediction_high = None
    if include_prediction_interval and model.strip().lower() == "random" and len(effects) > 1:
        pred_critical = approx_t_critical(max(len(effects) - 2, 1))
        pred_se = sqrt((ci_se ** 2) + base_result.tau_squared)
        prediction_low = base_result.pooled_effect - (pred_critical * pred_se)
        prediction_high = base_result.pooled_effect + (pred_critical * pred_se)

    return {
        "model": base_result.model,
        "ci_method": ci_method_name,
        "k": base_result.k,
        "pooled_effect": base_result.pooled_effect,
        "standard_error": conventional_se,
        "ci_standard_error": ci_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "tau_squared": base_result.tau_squared,
        "tau_method": base_result.tau_method,
        "q": base_result.q,
        "q_p_value": chi_square_sf(base_result.q, max(base_result.k - 1, 1)),
        "i_squared": base_result.i_squared,
        "h": base_result.h,
        "h_squared": base_result.h_squared,
        "prediction_low": prediction_low,
        "prediction_high": prediction_high,
    }


def leave_one_out_analysis(
    effects: list[StudyEffect],
    *,
    model: str = "random",
    ci_method: str = "normal",
    tau_method: str = "dl",
) -> list[dict[str, float | int | str | None]]:
    results: list[dict[str, float | int | str | None]] = []
    for index, effect in enumerate(effects):
        remaining = effects[:index] + effects[index + 1 :]
        if len(remaining) < 2:
            continue
        summary = summarize_effects(
            remaining,
            model=model,
            ci_method=ci_method,
            tau_method=tau_method,
        )
        results.append(
            {
                "excluded_study_id": effect.study_id,
                "excluded_label": effect.label,
                **summary,
            }
        )
    return results


def egger_test(effects: list[StudyEffect]) -> dict[str, float | int] | None:
    if len(effects) < 3:
        return None

    precisions = [1.0 / sqrt(effect.variance) for effect in effects]
    snd = [effect.effect_size / sqrt(effect.variance) for effect in effects]
    n = len(effects)
    mean_x = sum(precisions) / n
    mean_y = sum(snd) / n
    ss_xx = sum((x - mean_x) ** 2 for x in precisions)
    if ss_xx == 0:
        return None
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(precisions, snd))
    slope = ss_xy / ss_xx
    intercept = mean_y - (slope * mean_x)
    residual_sum_squares = sum((y - intercept - (slope * x)) ** 2 for x, y in zip(precisions, snd))
    sigma_squared = residual_sum_squares / max(n - 2, 1)
    intercept_se = sqrt(sigma_squared * ((1.0 / n) + (mean_x ** 2 / ss_xx)))
    slope_se = sqrt(sigma_squared / ss_xx)
    if intercept_se == 0 or slope_se == 0:
        return None

    intercept_z = intercept / intercept_se
    slope_z = slope / slope_se
    return {
        "k": n,
        "intercept": intercept,
        "intercept_se": intercept_se,
        "intercept_z": intercept_z,
        "intercept_p_value": normal_two_sided_p(intercept_z),
        "slope": slope,
        "slope_se": slope_se,
        "slope_z": slope_z,
        "slope_p_value": normal_two_sided_p(slope_z),
    }


def subgroup_analysis(
    effects: list[StudyEffect],
    metadata_key: str,
    *,
    model: str = "random",
    ci_method: str = "normal",
    tau_method: str = "dl",
) -> dict[str, Any]:
    groups: dict[str, list[StudyEffect]] = {}
    for effect in effects:
        subgroup = str(effect.metadata.get(metadata_key, "missing")).strip() or "missing"
        groups.setdefault(subgroup, []).append(effect)

    ordered_groups = dict(sorted(groups.items(), key=lambda item: item[0]))
    summaries: list[dict[str, Any]] = []
    for subgroup, subgroup_effects in ordered_groups.items():
        if len(subgroup_effects) >= 2:
            summary = summarize_effects(
                subgroup_effects,
                model=model,
                ci_method=ci_method,
                tau_method=tau_method,
            )
        else:
            summary = _single_study_summary(subgroup_effects[0], ci_method=ci_method)
        summaries.append(
            {
                "subgroup": subgroup,
                "study_ids": [effect.study_id for effect in subgroup_effects],
                **summary,
            }
        )

    total_fixed = summarize_effects(effects, model="fixed", ci_method="normal")
    within_q = sum(
        summarize_effects(subgroup_effects, model="fixed", ci_method="normal")["q"]
        if len(subgroup_effects) >= 2
        else 0.0
        for subgroup_effects in ordered_groups.values()
    )
    q_between = max(0.0, float(total_fixed["q"]) - within_q)
    df_between = max(len(ordered_groups) - 1, 1)
    return {
        "metadata_key": metadata_key,
        "group_count": len(ordered_groups),
        "q_between": q_between,
        "df_between": df_between,
        "p_value": chi_square_sf(q_between, df_between),
        "groups": summaries,
    }


def _sortable_metadata_value(value: Any) -> tuple[int, float | str]:
    if isinstance(value, bool):
        return 0, float(int(value))
    if isinstance(value, (int, float)):
        return 0, float(value)
    text = str(value or "").strip()
    try:
        return 0, float(text)
    except ValueError:
        return 1, text.lower()


def cumulative_meta_analysis(
    effects: list[StudyEffect],
    order_key: str,
    *,
    model: str = "random",
    ci_method: str = "normal",
    tau_method: str = "dl",
) -> list[dict[str, Any]]:
    ordered = sorted(
        effects,
        key=lambda effect: (_sortable_metadata_value(effect.metadata.get(order_key)), effect.study_id),
    )
    cumulative: list[dict[str, Any]] = []
    for end_index in range(1, len(ordered) + 1):
        current = ordered[:end_index]
        if len(current) == 1:
            summary = _single_study_summary(current[0], ci_method=ci_method)
        else:
            summary = summarize_effects(
                current,
                model=model,
                ci_method=ci_method,
                tau_method=tau_method,
            )
        cumulative.append(
            {
                "order_key": order_key,
                "order_value": current[-1].metadata.get(order_key),
                "study_id": current[-1].study_id,
                "included_study_ids": [effect.study_id for effect in current],
                **summary,
            }
        )
    return cumulative


def _coerce_covariate_keys(covariate_keys: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(covariate_keys, str):
        candidates = covariate_keys.split(",")
    else:
        candidates = list(covariate_keys)
    return [candidate.strip() for candidate in candidates if candidate and candidate.strip()]


def _try_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = [
        row[:] + [1.0 if row_index == column_index else 0.0 for column_index in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    for column_index in range(size):
        pivot_index = max(
            range(column_index, size),
            key=lambda row_index: abs(augmented[row_index][column_index]),
        )
        pivot_value = augmented[pivot_index][column_index]
        if abs(pivot_value) < SINGULARITY_EPS:
            raise ValueError("matrix is singular")
        if pivot_index != column_index:
            augmented[column_index], augmented[pivot_index] = augmented[pivot_index], augmented[column_index]

        pivot_row = augmented[column_index]
        scale = pivot_row[column_index]
        augmented[column_index] = [value / scale for value in pivot_row]

        for row_index in range(size):
            if row_index == column_index:
                continue
            factor = augmented[row_index][column_index]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                value - (factor * pivot)
                for value, pivot in zip(augmented[row_index], augmented[column_index])
            ]
    return [row[size:] for row in augmented]


def meta_regression(
    effects: list[StudyEffect],
    covariate_keys: str | list[str] | tuple[str, ...],
    *,
    model: str = "random",
    tau_method: str = "dl",
) -> dict[str, Any] | None:
    keys = _coerce_covariate_keys(covariate_keys)
    if not keys:
        return None

    covariate_rows: list[tuple[StudyEffect, list[float]]] = []
    for effect in effects:
        values: list[float] = []
        for key in keys:
            numeric = _try_float(effect.metadata.get(key))
            if numeric is None:
                values = []
                break
            values.append(numeric)
        if values:
            covariate_rows.append((effect, values))

    if len(covariate_rows) < len(keys) + 2:
        return None

    filtered_effects = [item[0] for item in covariate_rows]
    covariates = [item[1] for item in covariate_rows]
    base_result = run_meta_analysis(filtered_effects, model=model, tau_method=tau_method)
    tau_squared = base_result.tau_squared if model.strip().lower() == "random" else 0.0
    weights = [1.0 / (effect.variance + tau_squared) for effect in filtered_effects]
    y_values = [effect.effect_size for effect in filtered_effects]
    design = [[1.0, *row] for row in covariates]
    parameter_count = len(design[0])

    xtwx = [[0.0 for _ in range(parameter_count)] for _ in range(parameter_count)]
    xtwy = [0.0 for _ in range(parameter_count)]
    for row, y_value, weight in zip(design, y_values, weights):
        for column_index in range(parameter_count):
            xtwy[column_index] += weight * row[column_index] * y_value
            for inner_index in range(parameter_count):
                xtwx[column_index][inner_index] += weight * row[column_index] * row[inner_index]

    try:
        covariance = _invert_matrix(xtwx)
    except ValueError:
        return None

    coefficients = [
        sum(covariance[row_index][column_index] * xtwy[column_index] for column_index in range(parameter_count))
        for row_index in range(parameter_count)
    ]
    fitted_values = [
        sum(coefficient * value for coefficient, value in zip(coefficients, row))
        for row in design
    ]
    residual_q = sum(
        weight * ((observed - fitted) ** 2)
        for observed, fitted, weight in zip(y_values, fitted_values, weights)
    )
    weighted_mean = sum(weight * value for weight, value in zip(weights, y_values)) / sum(weights)
    total_q = sum(weight * ((value - weighted_mean) ** 2) for value, weight in zip(y_values, weights))
    moderator_q = max(0.0, total_q - residual_q)
    moderator_df = len(keys)

    coefficient_terms = ["intercept", *keys]
    coefficient_results: list[dict[str, float | str]] = []
    for row_index, (term, estimate, row) in enumerate(zip(coefficient_terms, coefficients, covariance)):
        standard_error = sqrt(max(row[row_index], 0.0))
        z_value = estimate / standard_error if standard_error > 0 else 0.0
        coefficient_results.append(
            {
                "term": term,
                "estimate": estimate,
                "standard_error": standard_error,
                "z": z_value,
                "p_value": normal_two_sided_p(z_value),
            }
        )

    result: dict[str, Any] = {
        "covariate_keys": keys,
        "model": model,
        "tau_method": base_result.tau_method,
        "k": len(filtered_effects),
        "tau_squared": base_result.tau_squared,
        "residual_q": residual_q,
        "moderator_q": moderator_q,
        "moderator_df": moderator_df,
        "moderator_p_value": chi_square_sf(moderator_q, moderator_df),
        "coefficients": coefficient_results,
    }
    if len(keys) == 1:
        intercept_result = coefficient_results[0]
        slope_result = coefficient_results[1]
        result.update(
            {
                "covariate_key": keys[0],
                "intercept": intercept_result["estimate"],
                "intercept_se": intercept_result["standard_error"],
                "intercept_z": intercept_result["z"],
                "intercept_p_value": intercept_result["p_value"],
                "slope": slope_result["estimate"],
                "slope_se": slope_result["standard_error"],
                "slope_z": slope_result["z"],
                "slope_p_value": slope_result["p_value"],
            }
        )
    return result


def _default_tau_upper(effects: list[StudyEffect]) -> float:
    effect_values = [effect.effect_size for effect in effects]
    mean_effect = sum(effect_values) / len(effect_values)
    dispersion = sum((value - mean_effect) ** 2 for value in effect_values) / max(len(effect_values) - 1, 1)
    max_standard_error = max(sqrt(effect.variance) for effect in effects)
    return max(0.25, 3.0 * sqrt(max(dispersion, 0.0) + (max_standard_error ** 2)))


def _tau_log_prior(tau: float, tau_prior: str, tau_scale: float, tau_max: float) -> float:
    prior_name = tau_prior.strip().lower()
    if tau < 0.0:
        return float("-inf")
    if prior_name == "half-normal":
        return log(sqrt(2.0 / pi) / tau_scale) - (0.5 * ((tau / tau_scale) ** 2))
    if prior_name == "half-cauchy":
        return log(2.0 / (pi * tau_scale * (1.0 + ((tau / tau_scale) ** 2))))
    if prior_name == "uniform":
        return -log(max(tau_max, SINGULARITY_EPS))
    raise ValueError("tau_prior must be 'half-normal', 'half-cauchy', or 'uniform'")


def bayesian_random_effects(
    effects: list[StudyEffect],
    *,
    mu_prior_mean: float = 0.0,
    mu_prior_sd: float = 1.0,
    tau_prior: str = "half-normal",
    tau_scale: float = 0.2,
    tau_max: float | None = None,
    grid_points: int = 301,
) -> dict[str, Any]:
    if len(effects) < 2:
        raise ValueError("at least two effects are required for Bayesian pooling")
    if mu_prior_sd <= 0.0:
        raise ValueError("mu_prior_sd must be positive")
    if tau_scale <= 0.0:
        raise ValueError("tau_scale must be positive")
    if grid_points < 25:
        raise ValueError("grid_points must be at least 25")

    tau_upper = tau_max if tau_max is not None else max(4.0 * tau_scale, _default_tau_upper(effects))
    tau_upper = max(tau_upper, tau_scale / 2.0)
    tau_grid = [tau_upper * index / (grid_points - 1) for index in range(grid_points)]
    prior_variance = mu_prior_sd ** 2
    prior_precision = 1.0 / prior_variance
    log_posterior_weights: list[float] = []
    mu_means: list[float] = []
    mu_variances: list[float] = []

    for tau in tau_grid:
        total_precision = prior_precision
        weighted_sum = mu_prior_mean * prior_precision
        constant_term = (mu_prior_mean ** 2) * prior_precision
        log_variance_term = log(prior_variance)
        for effect in effects:
            sampling_variance = effect.variance + (tau ** 2)
            precision = 1.0 / sampling_variance
            total_precision += precision
            weighted_sum += effect.effect_size * precision
            constant_term += (effect.effect_size ** 2) * precision
            log_variance_term += log(sampling_variance)

        posterior_variance = 1.0 / total_precision
        posterior_mean = posterior_variance * weighted_sum
        log_marginal = (
            -(len(effects) / 2.0) * LOG_2PI
            - 0.5 * log_variance_term
            - 0.5 * log(total_precision)
            - 0.5 * (constant_term - ((weighted_sum ** 2) / total_precision))
        )
        log_posterior_weights.append(log_marginal + _tau_log_prior(tau, tau_prior, tau_scale, tau_upper))
        mu_means.append(posterior_mean)
        mu_variances.append(posterior_variance)

    log_weight_total = _logsumexp(log_posterior_weights)
    tau_weights = [exp(value - log_weight_total) for value in log_posterior_weights]
    pooled_effect = sum(weight * mean for weight, mean in zip(tau_weights, mu_means))
    pooled_variance = sum(
        weight * (variance + (mean ** 2))
        for weight, variance, mean in zip(tau_weights, mu_variances, mu_means)
    ) - (pooled_effect ** 2)
    standard_error = sqrt(max(pooled_variance, 0.0))

    ci_low = _normal_mixture_quantile(mu_means, mu_variances, tau_weights, 0.025)
    ci_high = _normal_mixture_quantile(mu_means, mu_variances, tau_weights, 0.975)
    predictive_variances = [mu_variance + (tau ** 2) for mu_variance, tau in zip(mu_variances, tau_grid)]
    prediction_low = _normal_mixture_quantile(mu_means, predictive_variances, tau_weights, 0.025)
    prediction_high = _normal_mixture_quantile(mu_means, predictive_variances, tau_weights, 0.975)

    posterior_probability_negative = sum(
        weight * _normal_cdf(0.0, mean, variance)
        for weight, mean, variance in zip(tau_weights, mu_means, mu_variances)
    )
    tau_mean = sum(weight * tau for weight, tau in zip(tau_weights, tau_grid))
    tau_median = _weighted_grid_quantile(tau_grid, tau_weights, 0.5)
    tau_upper_95 = _weighted_grid_quantile(tau_grid, tau_weights, 0.95)

    return {
        "model": "bayesian-random-effects",
        "ci_method": "bayesian-credible",
        "k": len(effects),
        "pooled_effect": pooled_effect,
        "standard_error": standard_error,
        "ci_standard_error": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": prediction_low,
        "prediction_high": prediction_high,
        "tau_squared": tau_mean ** 2,
        "tau_method": "bayesian-grid",
        "tau_prior": tau_prior,
        "tau_scale": tau_scale,
        "tau_posterior_mean": tau_mean,
        "tau_posterior_median": tau_median,
        "tau_posterior_upper_95": tau_upper_95,
        "posterior_probability_negative": posterior_probability_negative,
        "posterior_probability_positive": 1.0 - posterior_probability_negative,
        "posterior_log_evidence": log_weight_total,
    }


def _cluster_identifier(effect: StudyEffect, cluster_key: str) -> str:
    key_name = cluster_key.strip()
    if key_name == "study_id":
        return effect.study_id
    if key_name == "label":
        return effect.label
    if key_name == "outcome":
        return effect.outcome
    if key_name == "comparator":
        return effect.comparator
    value = effect.metadata.get(key_name, "missing")
    text = str(value).strip()
    return text or "missing"


def _multilevel_cluster_terms(
    cluster_effects: list[StudyEffect],
    tau_between_squared: float,
    tau_within_squared: float,
) -> dict[str, float]:
    diagonal = [effect.variance + tau_within_squared for effect in cluster_effects]
    inverse_diagonal = [1.0 / value for value in diagonal]
    sum_inverse = sum(inverse_diagonal)
    sum_inverse_y = sum(weight * effect.effect_size for weight, effect in zip(inverse_diagonal, cluster_effects))
    sum_inverse_y2 = sum(weight * (effect.effect_size ** 2) for weight, effect in zip(inverse_diagonal, cluster_effects))
    denominator = 1.0 + (tau_between_squared * sum_inverse)
    information = sum_inverse / denominator
    weighted_sum = sum_inverse_y / denominator
    quadratic = sum_inverse_y2 - ((tau_between_squared * (sum_inverse_y ** 2)) / denominator)
    log_determinant = sum(log(value) for value in diagonal) + log(denominator)
    return {
        "information": information,
        "weighted_sum": weighted_sum,
        "quadratic": quadratic,
        "log_determinant": log_determinant,
        "sum_inverse": sum_inverse,
        "sum_inverse_y": sum_inverse_y,
        "denominator": denominator,
    }


def _fit_multilevel_given_taus(
    clustered_effects: dict[str, list[StudyEffect]],
    tau_between_squared: float,
    tau_within_squared: float,
) -> dict[str, Any]:
    cluster_terms = {
        cluster_id: _multilevel_cluster_terms(cluster_effects, tau_between_squared, tau_within_squared)
        for cluster_id, cluster_effects in clustered_effects.items()
    }
    total_information = sum(item["information"] for item in cluster_terms.values())
    if total_information <= 0.0:
        raise ValueError("non-positive total information in multilevel fit")
    pooled_effect = sum(item["weighted_sum"] for item in cluster_terms.values()) / total_information
    standard_error = sqrt(1.0 / total_information)
    log_likelihood = 0.0
    cluster_summaries: list[dict[str, Any]] = []
    for cluster_id, cluster_effects in clustered_effects.items():
        item = cluster_terms[cluster_id]
        residual_quadratic = item["quadratic"] - (2.0 * pooled_effect * item["weighted_sum"]) + (
            (pooled_effect ** 2) * item["information"]
        )
        log_likelihood += -0.5 * (
            (len(cluster_effects) * LOG_2PI)
            + item["log_determinant"]
            + residual_quadratic
        )
        cluster_random_effect_mean = (
            tau_between_squared * (item["sum_inverse_y"] - (pooled_effect * item["sum_inverse"])) / item["denominator"]
        )
        cluster_random_effect_variance = tau_between_squared / item["denominator"]
        cluster_summaries.append(
            {
                "cluster": cluster_id,
                "k": len(cluster_effects),
                "study_ids": sorted({effect.study_id for effect in cluster_effects}),
                "labels": [effect.label for effect in cluster_effects],
                "random_effect_mean": cluster_random_effect_mean,
                "random_effect_sd": sqrt(max(cluster_random_effect_variance, 0.0)),
                "cluster_effect": pooled_effect + cluster_random_effect_mean,
            }
        )
    return {
        "pooled_effect": pooled_effect,
        "standard_error": standard_error,
        "log_likelihood": log_likelihood,
        "clusters": cluster_summaries,
    }


def multilevel_meta_analysis(
    effects: list[StudyEffect],
    cluster_key: str = "study_id",
    *,
    grid_points: int = 21,
    refine_steps: int = 3,
    max_tau: float | None = None,
) -> dict[str, Any]:
    if len(effects) < 2:
        raise ValueError("at least two effects are required for multilevel meta-analysis")
    if grid_points < 5:
        raise ValueError("grid_points must be at least 5")
    if refine_steps < 1:
        raise ValueError("refine_steps must be at least 1")

    clustered_effects: dict[str, list[StudyEffect]] = {}
    for effect in effects:
        clustered_effects.setdefault(_cluster_identifier(effect, cluster_key), []).append(effect)
    if len(clustered_effects) < 2:
        raise ValueError("multilevel meta-analysis requires at least two clusters")

    tau_upper = max_tau if max_tau is not None else _default_tau_upper(effects)
    tau_upper = max(tau_upper, 0.1)
    between_low = 0.0
    between_high = tau_upper
    within_low = 0.0
    within_high = tau_upper
    best_between = 0.0
    best_within = 0.0
    best_fit: dict[str, Any] | None = None

    for _ in range(refine_steps):
        between_values = [
            between_low + ((between_high - between_low) * index / (grid_points - 1))
            for index in range(grid_points)
        ]
        within_values = [
            within_low + ((within_high - within_low) * index / (grid_points - 1))
            for index in range(grid_points)
        ]
        for tau_between in between_values:
            tau_between_squared = tau_between ** 2
            for tau_within in within_values:
                tau_within_squared = tau_within ** 2
                candidate = _fit_multilevel_given_taus(
                    clustered_effects,
                    tau_between_squared,
                    tau_within_squared,
                )
                if best_fit is None or candidate["log_likelihood"] > best_fit["log_likelihood"]:
                    best_between = tau_between
                    best_within = tau_within
                    best_fit = candidate
        assert best_fit is not None
        between_step = (between_high - between_low) / max(grid_points - 1, 1)
        within_step = (within_high - within_low) / max(grid_points - 1, 1)
        between_low = max(0.0, best_between - (2.0 * between_step))
        between_high = min(tau_upper, best_between + (2.0 * between_step))
        within_low = max(0.0, best_within - (2.0 * within_step))
        within_high = min(tau_upper, best_within + (2.0 * within_step))

    ci_low = best_fit["pooled_effect"] - (Z_975 * best_fit["standard_error"])
    ci_high = best_fit["pooled_effect"] + (Z_975 * best_fit["standard_error"])
    tau_between_squared = best_between ** 2
    tau_within_squared = best_within ** 2
    prediction_standard_error = sqrt(
        (best_fit["standard_error"] ** 2) + tau_between_squared + tau_within_squared
    )
    parameter_count = 3
    information_criterion_aic = -2.0 * best_fit["log_likelihood"] + (2.0 * parameter_count)
    information_criterion_bic = -2.0 * best_fit["log_likelihood"] + (log(len(effects)) * parameter_count)
    total_heterogeneity = tau_between_squared + tau_within_squared

    return {
        "model": "multilevel-random-effects",
        "ci_method": "normal",
        "k": len(effects),
        "cluster_key": cluster_key,
        "cluster_count": len(clustered_effects),
        "pooled_effect": best_fit["pooled_effect"],
        "standard_error": best_fit["standard_error"],
        "ci_standard_error": best_fit["standard_error"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "prediction_low": best_fit["pooled_effect"] - (Z_975 * prediction_standard_error),
        "prediction_high": best_fit["pooled_effect"] + (Z_975 * prediction_standard_error),
        "tau_squared": total_heterogeneity,
        "tau_method": "multilevel-grid",
        "tau_squared_between": tau_between_squared,
        "tau_squared_within": tau_within_squared,
        "intra_cluster_correlation": (
            0.0 if total_heterogeneity <= 0.0 else tau_between_squared / total_heterogeneity
        ),
        "log_likelihood": best_fit["log_likelihood"],
        "aic": information_criterion_aic,
        "bic": information_criterion_bic,
        "clusters": best_fit["clusters"],
    }


def format_advanced_summary(
    summary: dict[str, float | int | str | None],
    *,
    effect_scale: str,
) -> dict[str, float | int | str | None]:
    return {
        **summary,
        "pooled_estimate": back_transform(float(summary["pooled_effect"]), effect_scale),
        "estimate_ci_low": back_transform(float(summary["ci_low"]), effect_scale),
        "estimate_ci_high": back_transform(float(summary["ci_high"]), effect_scale),
        "estimate_prediction_low": (
            None
            if summary["prediction_low"] is None
            else back_transform(float(summary["prediction_low"]), effect_scale)
        ),
        "estimate_prediction_high": (
            None
            if summary["prediction_high"] is None
            else back_transform(float(summary["prediction_high"]), effect_scale)
        ),
    }
