from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Any

from .advanced_meta import chi_square_sf
from .models import NetworkContrast

NORMAL = NormalDist()
SINGULARITY_EPS = 1.0e-12
Z_975 = 1.959963984540054


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
            raise ValueError("network design matrix is singular")
        if pivot_index != column_index:
            augmented[column_index], augmented[pivot_index] = augmented[pivot_index], augmented[column_index]

        scale = augmented[column_index][column_index]
        augmented[column_index] = [value / scale for value in augmented[column_index]]
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


def _matrix_vector_multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(value * vector[column_index] for column_index, value in enumerate(row)) for row in matrix]


def _matrix_multiply(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(left[row_index][shared_index] * right[shared_index][column_index] for shared_index in range(len(right)))
            for column_index in range(len(right[0]))
        ]
        for row_index in range(len(left))
    ]


def _trace(matrix: list[list[float]]) -> float:
    return sum(matrix[index][index] for index in range(len(matrix)))


def _network_is_connected(contrasts: list[NetworkContrast], treatments: set[str]) -> bool:
    adjacency: dict[str, set[str]] = {treatment: set() for treatment in treatments}
    for contrast in contrasts:
        adjacency[contrast.treatment_a].add(contrast.treatment_b)
        adjacency[contrast.treatment_b].add(contrast.treatment_a)
    start = next(iter(treatments))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == treatments


def _parameter_mapping(treatments: list[str], reference_treatment: str) -> dict[str, int]:
    return {
        treatment: index
        for index, treatment in enumerate(treatment for treatment in treatments if treatment != reference_treatment)
    }


def _contrast_vector(contrast: NetworkContrast, mapping: dict[str, int], parameter_count: int) -> list[float]:
    vector = [0.0 for _ in range(parameter_count)]
    if contrast.treatment_a in mapping:
        vector[mapping[contrast.treatment_a]] += 1.0
    if contrast.treatment_b in mapping:
        vector[mapping[contrast.treatment_b]] -= 1.0
    return vector


def _fit_network_once(
    contrasts: list[NetworkContrast],
    treatments: list[str],
    reference_treatment: str,
    tau_squared: float,
) -> dict[str, Any]:
    mapping = _parameter_mapping(treatments, reference_treatment)
    parameter_count = len(mapping)
    if parameter_count == 0:
        raise ValueError("network meta-analysis requires at least two treatments")

    xtwx = [[0.0 for _ in range(parameter_count)] for _ in range(parameter_count)]
    xtw2x = [[0.0 for _ in range(parameter_count)] for _ in range(parameter_count)]
    xtwy = [0.0 for _ in range(parameter_count)]
    design_rows: list[list[float]] = []
    y_values: list[float] = []
    weights: list[float] = []

    for contrast in contrasts:
        row = _contrast_vector(contrast, mapping, parameter_count)
        weight = 1.0 / (contrast.variance + tau_squared)
        design_rows.append(row)
        y_values.append(contrast.effect_size)
        weights.append(weight)
        for row_index in range(parameter_count):
            xtwy[row_index] += weight * row[row_index] * contrast.effect_size
            for column_index in range(parameter_count):
                xtwx[row_index][column_index] += weight * row[row_index] * row[column_index]
                xtw2x[row_index][column_index] += (weight ** 2) * row[row_index] * row[column_index]

    covariance = _invert_matrix(xtwx)
    beta = _matrix_vector_multiply(covariance, xtwy)
    fitted_values = [
        sum(value * coefficient for value, coefficient in zip(row, beta))
        for row in design_rows
    ]
    residuals = [observed - fitted for observed, fitted in zip(y_values, fitted_values)]
    q = sum(weight * (residual ** 2) for weight, residual in zip(weights, residuals))
    c = sum(weights) - _trace(_matrix_multiply(covariance, xtw2x))
    return {
        "mapping": mapping,
        "covariance": covariance,
        "beta": beta,
        "q": q,
        "c": c,
        "weights": weights,
        "residuals": residuals,
    }


def _treatment_effect(
    treatment: str,
    reference_treatment: str,
    mapping: dict[str, int],
    beta: list[float],
) -> float:
    if treatment == reference_treatment:
        return 0.0
    return beta[mapping[treatment]]


def _pairwise_variance(
    treatment_a: str,
    treatment_b: str,
    reference_treatment: str,
    mapping: dict[str, int],
    covariance: list[list[float]],
) -> float:
    if treatment_a == treatment_b:
        return 0.0
    if treatment_a == reference_treatment:
        index = mapping[treatment_b]
        return covariance[index][index]
    if treatment_b == reference_treatment:
        index = mapping[treatment_a]
        return covariance[index][index]
    index_a = mapping[treatment_a]
    index_b = mapping[treatment_b]
    return (
        covariance[index_a][index_a]
        + covariance[index_b][index_b]
        - (2.0 * covariance[index_a][index_b])
    )


def network_meta_analysis(
    contrasts: list[NetworkContrast],
    *,
    reference_treatment: str = "",
    model: str = "random",
    smaller_effect_is_better: bool = True,
    tolerance: float = 1.0e-10,
    max_iter: int = 100,
) -> dict[str, Any]:
    if len(contrasts) < 2:
        raise ValueError("network meta-analysis requires at least two contrasts")

    treatments = sorted({contrast.treatment_a for contrast in contrasts} | {contrast.treatment_b for contrast in contrasts})
    if len(treatments) < 3:
        raise ValueError("network meta-analysis requires at least three treatments")
    if not _network_is_connected(contrasts, set(treatments)):
        raise ValueError("network contrasts must form a connected treatment graph")

    model_name = model.strip().lower()
    if model_name not in {"fixed", "random"}:
        raise ValueError("model must be 'fixed' or 'random'")

    reference = reference_treatment.strip() or treatments[0]
    if reference not in treatments:
        raise ValueError("reference_treatment must be present in the network")

    fit = _fit_network_once(contrasts, treatments, reference, 0.0)
    tau_squared = 0.0
    if model_name == "random":
        degrees_of_freedom = len(contrasts) - (len(treatments) - 1)
        for _ in range(max_iter):
            candidate = _fit_network_once(contrasts, treatments, reference, tau_squared)
            c_value = candidate["c"]
            if c_value <= 0.0:
                fit = candidate
                break
            updated_tau_squared = max(0.0, (candidate["q"] - degrees_of_freedom) / c_value)
            fit = candidate
            if abs(updated_tau_squared - tau_squared) <= tolerance:
                tau_squared = updated_tau_squared
                break
            tau_squared = updated_tau_squared
        fit = _fit_network_once(contrasts, treatments, reference, tau_squared)

    mapping = fit["mapping"]
    beta = fit["beta"]
    covariance = fit["covariance"]
    treatment_effects: list[dict[str, Any]] = []
    for treatment in treatments:
        effect = _treatment_effect(treatment, reference, mapping, beta)
        variance = 0.0 if treatment == reference else covariance[mapping[treatment]][mapping[treatment]]
        standard_error = 0.0 if treatment == reference else sqrt(max(variance, 0.0))
        treatment_effects.append(
            {
                "treatment": treatment,
                "reference_treatment": reference,
                "effect": effect,
                "standard_error": standard_error,
                "ci_low": effect - (Z_975 * standard_error),
                "ci_high": effect + (Z_975 * standard_error),
            }
        )

    pairwise_effects: list[dict[str, Any]] = []
    scores = {treatment: 0.0 for treatment in treatments}
    for left_index, treatment_a in enumerate(treatments):
        for treatment_b in treatments[left_index + 1 :]:
            effect = _treatment_effect(treatment_a, reference, mapping, beta) - _treatment_effect(treatment_b, reference, mapping, beta)
            variance = max(_pairwise_variance(treatment_a, treatment_b, reference, mapping, covariance), 0.0)
            standard_error = sqrt(max(variance, SINGULARITY_EPS))
            z_value = effect / standard_error
            probability_a_better = NORMAL.cdf((-effect) / standard_error) if smaller_effect_is_better else NORMAL.cdf(effect / standard_error)
            probability_b_better = 1.0 - probability_a_better
            scores[treatment_a] += probability_a_better
            scores[treatment_b] += probability_b_better
            pairwise_effects.append(
                {
                    "treatment_a": treatment_a,
                    "treatment_b": treatment_b,
                    "effect": effect,
                    "standard_error": standard_error,
                    "ci_low": effect - (Z_975 * standard_error),
                    "ci_high": effect + (Z_975 * standard_error),
                    "z": z_value,
                    "p_value": 2.0 * (1.0 - NORMAL.cdf(abs(z_value))),
                    "probability_treatment_a_better": probability_a_better,
                }
            )

    treatment_ranking = sorted(
        (
            {
                "treatment": treatment,
                "p_score": scores[treatment] / max(len(treatments) - 1, 1),
            }
            for treatment in treatments
        ),
        key=lambda item: item["p_score"],
        reverse=True,
    )
    degrees_of_freedom = max(len(contrasts) - (len(treatments) - 1), 1)
    return {
        "model": f"network-{model_name}",
        "k": len(contrasts),
        "treatment_count": len(treatments),
        "reference_treatment": reference,
        "tau_squared": tau_squared,
        "tau_method": "network-method-of-moments" if model_name == "random" else "fixed",
        "residual_q": fit["q"],
        "residual_df": degrees_of_freedom,
        "residual_q_p_value": chi_square_sf(fit["q"], degrees_of_freedom),
        "treatments": treatment_effects,
        "pairwise_effects": pairwise_effects,
        "ranking": treatment_ranking,
    }
