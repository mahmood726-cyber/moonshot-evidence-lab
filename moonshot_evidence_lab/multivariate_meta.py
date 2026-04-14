from __future__ import annotations

from math import sqrt
from typing import Any

from .advanced_meta import chi_square_sf
from .meta_analysis import Z_975, run_meta_analysis
from .models import MultivariateStudy, StudyEffect

SINGULARITY_EPS = 1.0e-12


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
            raise ValueError("multivariate covariance matrix is singular")
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


def _matrix_add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [left[row_index][column_index] + right[row_index][column_index] for column_index in range(len(left[row_index]))]
        for row_index in range(len(left))
    ]


def _validate_outcome_layout(studies: list[MultivariateStudy]) -> list[str]:
    outcomes = studies[0].outcomes
    for study in studies[1:]:
        if study.outcomes != outcomes:
            raise ValueError("all multivariate studies must share the same ordered outcome set")
    return outcomes


def _diagonal_tau_matrix(studies: list[MultivariateStudy], outcomes: list[str], model: str, tau_method: str) -> list[list[float]]:
    tau_values = [0.0 for _ in outcomes]
    if model == "random":
        for outcome_index, outcome in enumerate(outcomes):
            effects = [
                StudyEffect(
                    study_id=study.study_id,
                    label=study.label,
                    effect_size=study.effects[outcome],
                    variance=study.covariance[outcome_index][outcome_index],
                    outcome=outcome,
                )
                for study in studies
            ]
            tau_values[outcome_index] = run_meta_analysis(effects, model="random", tau_method=tau_method).tau_squared
    return [
        [tau_values[row_index] if row_index == column_index else 0.0 for column_index in range(len(outcomes))]
        for row_index in range(len(outcomes))
    ]


def _vector_subtract(left: list[float], right: list[float]) -> list[float]:
    return [left[index] - right[index] for index in range(len(left))]


def _quadratic_form(matrix: list[list[float]], vector: list[float]) -> float:
    product = _matrix_vector_multiply(matrix, vector)
    return sum(value * other for value, other in zip(vector, product))


def multivariate_meta_analysis(
    studies: list[MultivariateStudy],
    *,
    model: str = "random",
    tau_method: str = "pm",
) -> dict[str, Any]:
    if len(studies) < 2:
        raise ValueError("multivariate meta-analysis requires at least two studies")

    model_name = model.strip().lower()
    if model_name not in {"fixed", "random"}:
        raise ValueError("model must be 'fixed' or 'random'")
    outcomes = _validate_outcome_layout(studies)
    outcome_count = len(outcomes)
    tau_matrix = _diagonal_tau_matrix(studies, outcomes, model_name, tau_method)

    information = [[0.0 for _ in outcomes] for _ in outcomes]
    score = [0.0 for _ in outcomes]
    study_precision_matrices: list[list[list[float]]] = []
    study_vectors: list[list[float]] = []
    for study in studies:
        covariance = _matrix_add(study.covariance, tau_matrix)
        precision = _invert_matrix(covariance)
        vector = [study.effects[outcome] for outcome in outcomes]
        study_precision_matrices.append(precision)
        study_vectors.append(vector)
        weighted_vector = _matrix_vector_multiply(precision, vector)
        for row_index in range(outcome_count):
            score[row_index] += weighted_vector[row_index]
            for column_index in range(outcome_count):
                information[row_index][column_index] += precision[row_index][column_index]

    pooled_covariance = _invert_matrix(information)
    pooled_effects_vector = _matrix_vector_multiply(pooled_covariance, score)

    q = 0.0
    for precision, vector in zip(study_precision_matrices, study_vectors):
        residual = _vector_subtract(vector, pooled_effects_vector)
        q += _quadratic_form(precision, residual)

    residual_df = max((len(studies) * outcome_count) - outcome_count, 1)
    outcome_summaries: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(outcomes):
        pooled_effect = pooled_effects_vector[outcome_index]
        standard_error = sqrt(max(pooled_covariance[outcome_index][outcome_index], 0.0))
        tau_squared = tau_matrix[outcome_index][outcome_index]
        prediction_standard_error = sqrt((standard_error ** 2) + tau_squared)
        outcome_summaries.append(
            {
                "outcome": outcome,
                "pooled_effect": pooled_effect,
                "standard_error": standard_error,
                "ci_low": pooled_effect - (Z_975 * standard_error),
                "ci_high": pooled_effect + (Z_975 * standard_error),
                "prediction_low": pooled_effect - (Z_975 * prediction_standard_error),
                "prediction_high": pooled_effect + (Z_975 * prediction_standard_error),
                "tau_squared": tau_squared,
            }
        )

    pooled_correlation = [[0.0 for _ in outcomes] for _ in outcomes]
    for row_index in range(outcome_count):
        for column_index in range(outcome_count):
            row_sd = sqrt(max(pooled_covariance[row_index][row_index], 0.0))
            column_sd = sqrt(max(pooled_covariance[column_index][column_index], 0.0))
            denominator = row_sd * column_sd
            pooled_correlation[row_index][column_index] = (
                0.0 if denominator <= 0.0 else pooled_covariance[row_index][column_index] / denominator
            )

    return {
        "model": f"multivariate-{model_name}",
        "k": len(studies),
        "outcome_count": outcome_count,
        "outcomes": outcome_summaries,
        "pooled_covariance": pooled_covariance,
        "pooled_correlation": pooled_correlation,
        "tau_method": tau_method if model_name == "random" else "fixed",
        "between_study_covariance_assumption": "diagonal",
        "residual_q": q,
        "residual_df": residual_df,
        "residual_q_p_value": chi_square_sf(q, residual_df),
    }
