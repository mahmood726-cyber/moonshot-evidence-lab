from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .advanced_meta import (
    arcsine_proportion,
    log_ratio_of_means,
    logit_proportion,
    log_risk_ratio,
    mean_difference,
    raw_proportion,
    risk_difference,
    standardized_mean_difference,
)
from .meta_analysis import log_odds_ratio
from .models import MultivariateStudy, NetworkContrast, StudyEffect


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(str(value).replace(",", ""))


def _optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(str(value).replace(",", "")))


def _coerce_metadata(value: str | None) -> Any:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _base_metadata(row: dict[str, str], excluded: set[str]) -> dict[str, Any]:
    return {key: _coerce_metadata(value) for key, value in row.items() if key not in excluded and value not in (None, "")}


def load_effects_csv(path: str | Path) -> list[StudyEffect]:
    rows = _read_rows(path)
    effects: list[StudyEffect] = []
    excluded = {"study_id", "label", "effect_size", "variance", "outcome", "comparator"}
    for row in rows:
        metadata = _base_metadata(row, excluded)
        effects.append(
            StudyEffect(
                study_id=row["study_id"],
                label=row.get("label", "") or row["study_id"],
                effect_size=float(row["effect_size"]),
                variance=float(row["variance"]),
                outcome=row.get("outcome", "") or "",
                comparator=row.get("comparator", "") or "",
                metadata=metadata,
            )
        )
    return effects


def load_network_contrast_csv(path: str | Path) -> list[NetworkContrast]:
    rows = _read_rows(path)
    contrasts: list[NetworkContrast] = []
    excluded = {
        "study_id",
        "label",
        "treatment_a",
        "treatment_b",
        "effect_size",
        "variance",
        "outcome",
    }
    for row in rows:
        metadata = _base_metadata(row, excluded)
        contrasts.append(
            NetworkContrast(
                study_id=row["study_id"],
                label=row.get("label", "") or row["study_id"],
                treatment_a=row["treatment_a"],
                treatment_b=row["treatment_b"],
                effect_size=float(row["effect_size"]),
                variance=float(row["variance"]),
                outcome=row.get("outcome", "") or "",
                metadata=metadata,
            )
        )
    return contrasts


def load_multivariate_wide_csv(path: str | Path) -> list[MultivariateStudy]:
    rows = _read_rows(path)
    if not rows:
        return []

    header = list(rows[0].keys())
    effect_columns = [column for column in header if column.endswith("__effect")]
    outcomes = [column.removesuffix("__effect") for column in effect_columns]
    if not outcomes:
        raise ValueError("multivariate wide CSV requires at least one '<outcome>__effect' column")

    variance_columns = {f"{outcome}__variance" for outcome in outcomes}
    covariance_columns = {
        key
        for key in header
        if key.endswith("__covariance") and key.count("__") >= 2
    }
    excluded = {
        "study_id",
        "label",
        *effect_columns,
        *variance_columns,
        *covariance_columns,
    }
    studies: list[MultivariateStudy] = []
    for row in rows:
        effects = {
            outcome: float(row[f"{outcome}__effect"])
            for outcome in outcomes
        }
        covariance = [[0.0 for _ in outcomes] for _ in outcomes]
        for row_index, outcome in enumerate(outcomes):
            variance_key = f"{outcome}__variance"
            if row.get(variance_key, "") in (None, ""):
                raise ValueError(f"missing variance column '{variance_key}'")
            covariance[row_index][row_index] = float(row[variance_key])
            for column_index in range(row_index + 1, len(outcomes)):
                other_outcome = outcomes[column_index]
                direct_key = f"{outcome}__{other_outcome}__covariance"
                reverse_key = f"{other_outcome}__{outcome}__covariance"
                covariance_value = _optional_float(row.get(direct_key))
                if covariance_value is None:
                    covariance_value = _optional_float(row.get(reverse_key))
                covariance[row_index][column_index] = covariance_value or 0.0
                covariance[column_index][row_index] = covariance[row_index][column_index]

        metadata = _base_metadata(row, excluded)
        studies.append(
            MultivariateStudy(
                study_id=row["study_id"],
                label=row.get("label", "") or row["study_id"],
                outcomes=list(outcomes),
                effects=effects,
                covariance=covariance,
                metadata=metadata,
            )
        )
    return studies


def load_binary_arms_csv(path: str | Path, *, measure: str = "or") -> list[StudyEffect]:
    measure_name = measure.strip().lower()
    rows = _read_rows(path)
    effects: list[StudyEffect] = []
    excluded = {
        "study_id",
        "label",
        "outcome",
        "comparator",
        "treatment_events",
        "treatment_total",
        "control_events",
        "control_total",
        "treatment_label",
        "control_label",
    }
    for row in rows:
        treatment_events = int(row["treatment_events"])
        treatment_total = int(row["treatment_total"])
        control_events = int(row["control_events"])
        control_total = int(row["control_total"])

        if measure_name == "or":
            effect_size, variance = log_odds_ratio(
                treatment_events,
                treatment_total - treatment_events,
                control_events,
                control_total - control_events,
            )
            metric = "or"
            effect_scale = "log"
        elif measure_name == "rr":
            effect_size, variance = log_risk_ratio(
                treatment_events,
                treatment_total,
                control_events,
                control_total,
            )
            metric = "rr"
            effect_scale = "log"
        elif measure_name == "rd":
            effect_size, variance = risk_difference(
                treatment_events,
                treatment_total,
                control_events,
                control_total,
            )
            metric = "risk_difference"
            effect_scale = "identity"
        else:
            raise ValueError("binary measure must be one of: or, rr, rd")

        metadata = _base_metadata(row, excluded)
        metadata.update(
            {
                "source": metadata.get("source", "local_csv"),
                "metric": metric,
                "effect_scale": effect_scale,
                "sample_size": treatment_total + control_total,
                "year": _optional_int(row.get("year")),
                "treatment_label": row.get("treatment_label", "") or "",
                "control_label": row.get("control_label", "") or "",
            }
        )
        comparator = row.get("comparator", "") or " vs ".join(
            part for part in (row.get("treatment_label", ""), row.get("control_label", "")) if part
        )
        effects.append(
            StudyEffect(
                study_id=row["study_id"],
                label=row.get("label", "") or row["study_id"],
                effect_size=effect_size,
                variance=variance,
                outcome=row.get("outcome", "") or "",
                comparator=comparator,
                metadata=metadata,
            )
        )
    return effects


def load_continuous_arms_csv(path: str | Path, *, measure: str = "md") -> list[StudyEffect]:
    measure_name = measure.strip().lower()
    rows = _read_rows(path)
    effects: list[StudyEffect] = []
    excluded = {
        "study_id",
        "label",
        "outcome",
        "comparator",
        "treatment_mean",
        "treatment_sd",
        "treatment_n",
        "control_mean",
        "control_sd",
        "control_n",
        "treatment_label",
        "control_label",
    }
    for row in rows:
        treatment_mean = float(row["treatment_mean"])
        treatment_sd = float(row["treatment_sd"])
        treatment_n = int(row["treatment_n"])
        control_mean = float(row["control_mean"])
        control_sd = float(row["control_sd"])
        control_n = int(row["control_n"])

        if measure_name == "md":
            effect_size, variance = mean_difference(
                treatment_mean,
                treatment_sd,
                treatment_n,
                control_mean,
                control_sd,
                control_n,
            )
            metric = "mean_difference"
            effect_scale = "identity"
        elif measure_name == "smd":
            effect_size, variance = standardized_mean_difference(
                treatment_mean,
                treatment_sd,
                treatment_n,
                control_mean,
                control_sd,
                control_n,
            )
            metric = "smd"
            effect_scale = "identity"
        elif measure_name == "rom":
            effect_size, variance = log_ratio_of_means(
                treatment_mean,
                treatment_sd,
                treatment_n,
                control_mean,
                control_sd,
                control_n,
            )
            metric = "rom"
            effect_scale = "log"
        else:
            raise ValueError("continuous measure must be one of: md, smd, rom")

        metadata = _base_metadata(row, excluded)
        metadata.update(
            {
                "source": metadata.get("source", "local_csv"),
                "metric": metric,
                "effect_scale": effect_scale,
                "sample_size": treatment_n + control_n,
                "year": _optional_int(row.get("year")),
                "treatment_label": row.get("treatment_label", "") or "",
                "control_label": row.get("control_label", "") or "",
            }
        )
        comparator = row.get("comparator", "") or " vs ".join(
            part for part in (row.get("treatment_label", ""), row.get("control_label", "")) if part
        )
        effects.append(
            StudyEffect(
                study_id=row["study_id"],
                label=row.get("label", "") or row["study_id"],
                effect_size=effect_size,
                variance=variance,
                outcome=row.get("outcome", "") or "",
                comparator=comparator,
                metadata=metadata,
            )
        )
    return effects


def load_proportion_csv(path: str | Path, *, measure: str = "logit") -> list[StudyEffect]:
    measure_name = measure.strip().lower()
    rows = _read_rows(path)
    effects: list[StudyEffect] = []
    excluded = {
        "study_id",
        "label",
        "outcome",
        "events",
        "total",
        "comparator",
    }
    for row in rows:
        events = int(row["events"])
        total = int(row["total"])

        if measure_name == "logit":
            effect_size, variance = logit_proportion(events, total)
            metric = "proportion"
            effect_scale = "logit"
        elif measure_name == "raw":
            effect_size, variance = raw_proportion(events, total)
            metric = "proportion"
            effect_scale = "identity"
        elif measure_name == "arcsine":
            effect_size, variance = arcsine_proportion(events, total)
            metric = "proportion"
            effect_scale = "arcsine"
        else:
            raise ValueError("proportion measure must be one of: logit, raw, arcsine")

        metadata = _base_metadata(row, excluded)
        metadata.update(
            {
                "source": metadata.get("source", "local_csv"),
                "metric": metric,
                "effect_scale": effect_scale,
                "sample_size": total,
                "events": events,
                "year": _optional_int(row.get("year")),
            }
        )
        effects.append(
            StudyEffect(
                study_id=row["study_id"],
                label=row.get("label", "") or row["study_id"],
                effect_size=effect_size,
                variance=variance,
                outcome=row.get("outcome", "") or "",
                comparator=row.get("comparator", "") or "",
                metadata=metadata,
            )
        )
    return effects


def load_effects_dataset(
    path: str | Path,
    *,
    dataset_format: str,
    measure: str = "",
) -> list[StudyEffect]:
    format_name = dataset_format.strip().lower()
    if format_name == "effects":
        return load_effects_csv(path)
    if format_name == "binary":
        return load_binary_arms_csv(path, measure=measure or "or")
    if format_name == "continuous":
        return load_continuous_arms_csv(path, measure=measure or "md")
    if format_name == "proportion":
        return load_proportion_csv(path, measure=measure or "logit")
    raise ValueError("dataset_format must be one of: effects, binary, continuous, proportion")
