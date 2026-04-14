from __future__ import annotations

import re
from math import log
from typing import Any

from .meta_analysis import Z_975, log_odds_ratio
from .models import ExtractedEffectCandidate

POSITIVE_NUMBER_PATTERN = r"\d+(?:\.\d+)?"
SIGNED_NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"

RATIO_CI_PATTERN = re.compile(
    rf"\b(?P<metric>odds ratio|risk ratio|relative risk|hazard ratio|OR|RR|HR)\b"
    rf"[^\d]{{0,20}}(?P<estimate>{POSITIVE_NUMBER_PATTERN})"
    rf"[^\d]{{0,40}}(?:95%\s*CI|CI)"
    rf"[^\d]{{0,12}}(?P<low>{POSITIVE_NUMBER_PATTERN})\s*(?:to|-|–|—)\s*(?P<high>{POSITIVE_NUMBER_PATTERN})",
    re.IGNORECASE,
)

ADDITIVE_CI_PATTERN = re.compile(
    rf"\b(?P<metric>mean difference|difference in means|risk difference|rate difference)\b"
    rf"[^\d\-]{{0,20}}(?P<estimate>{SIGNED_NUMBER_PATTERN})"
    rf"[^\d]{{0,40}}(?:95%\s*CI|CI)"
    rf"[^\d\-]{{0,12}}(?P<low>{SIGNED_NUMBER_PATTERN})\s*(?:to|-|–|—)\s*(?P<high>{SIGNED_NUMBER_PATTERN})",
    re.IGNORECASE,
)

BINARY_COUNTS_PATTERN = re.compile(
    r"\b(?P<t_events>\d+)\s*/\s*(?P<t_total>\d+)\s*(?:vs\.?|versus)\s*(?P<c_events>\d+)\s*/\s*(?P<c_total>\d+)\b",
    re.IGNORECASE,
)


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _snippet(text: str, start: int, end: int, *, context: int = 80) -> str:
    left = max(0, start - context)
    right = min(len(text), end + context)
    return " ".join(text[left:right].split())


def normalize_metric_name(metric_text: str) -> str:
    normalized = metric_text.strip().lower()
    if normalized in {"or", "odds ratio"}:
        return "or"
    if normalized in {"rr", "risk ratio", "relative risk"}:
        return "rr"
    if normalized in {"hr", "hazard ratio"}:
        return "hr"
    if normalized == "mean difference":
        return "mean_difference"
    if normalized == "difference in means":
        return "mean_difference"
    if normalized == "risk difference":
        return "risk_difference"
    if normalized == "rate difference":
        return "rate_difference"
    return normalized.replace(" ", "_")


def effect_scale_for_metric(metric: str) -> str:
    return "log" if metric in {"or", "rr", "hr"} else "identity"


def variance_from_confidence_interval(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    metric: str,
) -> tuple[float, float]:
    if ci_low >= ci_high:
        raise ValueError("confidence interval lower bound must be less than upper bound")

    if effect_scale_for_metric(metric) == "log":
        if min(estimate, ci_low, ci_high) <= 0:
            raise ValueError("ratio estimates and bounds must be positive")
        effect_size = log(estimate)
        standard_error = (log(ci_high) - log(ci_low)) / (2.0 * Z_975)
    else:
        effect_size = estimate
        standard_error = (ci_high - ci_low) / (2.0 * Z_975)

    variance = standard_error ** 2
    if variance <= 0:
        raise ValueError("variance must be positive")
    return effect_size, variance


def candidate_from_estimate_and_ci(
    *,
    source_record_id: str,
    source: str,
    metric: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    outcome: str = "",
    comparator: str = "",
    snippet: str = "",
    extraction_method: str = "",
    metadata: dict[str, Any] | None = None,
) -> ExtractedEffectCandidate:
    normalized_metric = normalize_metric_name(metric)
    effect_size, variance = variance_from_confidence_interval(
        estimate,
        ci_low,
        ci_high,
        metric=normalized_metric,
    )
    return ExtractedEffectCandidate(
        source_record_id=source_record_id,
        source=source,
        metric=normalized_metric,
        effect_size=effect_size,
        variance=variance,
        effect_scale=effect_scale_for_metric(normalized_metric),
        outcome=outcome,
        comparator=comparator,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        snippet=snippet,
        extraction_method=extraction_method,
        metadata=metadata or {},
    )


def extract_effect_candidates_from_text(
    text: str,
    *,
    source_record_id: str,
    source: str,
    outcome: str = "",
    comparator: str = "",
) -> list[ExtractedEffectCandidate]:
    candidates: list[ExtractedEffectCandidate] = []
    if not text.strip():
        return candidates

    seen: set[tuple[str, int, int]] = set()

    for pattern, method_name in (
        (RATIO_CI_PATTERN, "text_ratio_ci"),
        (ADDITIVE_CI_PATTERN, "text_additive_ci"),
    ):
        for match in pattern.finditer(text):
            key = (method_name, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)

            estimate = float(match.group("estimate"))
            ci_low = float(match.group("low"))
            ci_high = float(match.group("high"))
            metric = match.group("metric")
            candidates.append(
                candidate_from_estimate_and_ci(
                    source_record_id=source_record_id,
                    source=source,
                    metric=metric,
                    estimate=estimate,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    outcome=outcome,
                    comparator=comparator,
                    snippet=_snippet(text, match.start(), match.end()),
                    extraction_method=method_name,
                )
            )

    for match in BINARY_COUNTS_PATTERN.finditer(text):
        t_events = int(match.group("t_events"))
        t_total = int(match.group("t_total"))
        c_events = int(match.group("c_events"))
        c_total = int(match.group("c_total"))
        if t_total <= t_events or c_total <= c_events:
            continue
        effect_size, variance = log_odds_ratio(
            t_events,
            t_total - t_events,
            c_events,
            c_total - c_events,
        )
        candidates.append(
            ExtractedEffectCandidate(
                source_record_id=source_record_id,
                source=source,
                metric="or",
                effect_size=effect_size,
                variance=variance,
                effect_scale="log",
                outcome=outcome,
                comparator=comparator,
                snippet=_snippet(text, match.start(), match.end()),
                extraction_method="text_binary_counts",
                metadata={
                    "treatment_events": t_events,
                    "treatment_total": t_total,
                    "control_events": c_events,
                    "control_total": c_total,
                },
            )
        )

    return candidates


def _coerce_study_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("study"), dict):
        return payload["study"]
    studies = payload.get("studies")
    if isinstance(studies, list) and studies:
        first = studies[0]
        if isinstance(first, dict):
            return first
    return payload


def _string_value(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _collect_ctgov_group_titles(study: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def add_groups(groups: Any) -> None:
        if not isinstance(groups, list):
            return
        for group in groups:
            if not isinstance(group, dict):
                continue
            group_id = _string_value(group.get("id"), group.get("groupId"))
            title = _string_value(
                group.get("title"),
                group.get("label"),
                group.get("armGroupLabel"),
                group.get("description"),
            )
            if group_id and title:
                mapping[group_id] = title

    protocol = study.get("protocolSection", {})
    if isinstance(protocol, dict):
        arms = protocol.get("armsInterventionsModule", {})
        if isinstance(arms, dict):
            add_groups(arms.get("armGroups"))

    results = study.get("resultsSection", {})
    if isinstance(results, dict):
        participant_flow = results.get("participantFlowModule", {})
        baseline = results.get("baselineCharacteristicsModule", {})
        if isinstance(participant_flow, dict):
            add_groups(participant_flow.get("groups"))
        if isinstance(baseline, dict):
            add_groups(baseline.get("groups"))

    return mapping


def _group_titles_from_group_list(groups: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not isinstance(groups, list):
        return mapping
    for group in groups:
        if not isinstance(group, dict):
            continue
        group_id = _string_value(group.get("id"), group.get("groupId"))
        title = _string_value(group.get("title"), group.get("label"), group.get("description"))
        if group_id and title:
            mapping[group_id] = title
    return mapping


def _extract_group_ids(groups: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(groups, list):
        return ids
    for group in groups:
        if isinstance(group, dict):
            group_id = _string_value(group.get("id"), group.get("groupId"))
        else:
            group_id = _string_value(group)
        if group_id:
            ids.append(group_id)
    return ids


def _comparator_from_group_ids(group_ids: list[str], titles: dict[str, str]) -> str:
    if not group_ids:
        return ""
    labels = [titles.get(group_id, group_id) for group_id in group_ids]
    return " vs ".join(labels)


def _is_control_group(title: str) -> bool:
    lowered = title.strip().lower()
    return any(token in lowered for token in ("placebo", "control", "usual care", "standard care", "sham"))


def _ordered_pair(group_a: str, group_b: str, titles: dict[str, str]) -> tuple[str, str]:
    title_a = titles.get(group_a, group_a)
    title_b = titles.get(group_b, group_b)
    if _is_control_group(title_a) and not _is_control_group(title_b):
        return group_b, group_a
    if _is_control_group(title_b) and not _is_control_group(title_a):
        return group_a, group_b
    return group_a, group_b


def _iter_measurement_sets(outcome: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    sets: list[tuple[str, list[dict[str, Any]]]] = []

    classes = outcome.get("classes")
    if isinstance(classes, list):
        for outcome_class in classes:
            if not isinstance(outcome_class, dict):
                continue
            class_title = _string_value(outcome_class.get("title"))
            categories = outcome_class.get("categories")
            if isinstance(categories, list):
                for category in categories:
                    if not isinstance(category, dict):
                        continue
                    category_title = _string_value(category.get("title"))
                    label = " / ".join(part for part in (class_title, category_title) if part)
                    measurements = category.get("measurements")
                    if isinstance(measurements, list):
                        sets.append((label, measurements))
            else:
                measurements = outcome_class.get("measurements")
                if isinstance(measurements, list):
                    sets.append((class_title, measurements))

    categories = outcome.get("categories")
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, dict):
                continue
            category_title = _string_value(category.get("title"))
            measurements = category.get("measurements")
            if isinstance(measurements, list):
                sets.append((category_title, measurements))

    measurements = outcome.get("measurements")
    if isinstance(measurements, list):
        sets.append(("", measurements))

    return sets


def _extract_denominators(outcome: dict[str, Any]) -> dict[str, float]:
    denoms: dict[str, float] = {}
    raw_denoms = outcome.get("denoms")
    if not isinstance(raw_denoms, list):
        return denoms
    for denom in raw_denoms:
        if not isinstance(denom, dict):
            continue
        group_id = _string_value(denom.get("groupId"), denom.get("id"))
        value = _parse_float(denom.get("value"))
        if group_id and value is not None:
            denoms[group_id] = value
            continue
        counts = denom.get("counts")
        if not isinstance(counts, list):
            continue
        for count in counts:
            if not isinstance(count, dict):
                continue
            count_group_id = _string_value(count.get("groupId"), count.get("id"))
            count_value = _parse_float(count.get("value"))
            if count_group_id and count_value is not None:
                denoms[count_group_id] = count_value
    return denoms


def _infer_ctgov_metric(text: str) -> str | None:
    normalized = normalize_metric_name(text)
    if normalized in {"or", "rr", "hr", "mean_difference", "risk_difference", "rate_difference"}:
        return normalized

    lowered = text.strip().lower()
    if "odds ratio" in lowered:
        return "or"
    if "risk ratio" in lowered or "relative risk" in lowered:
        return "rr"
    if "hazard ratio" in lowered:
        return "hr"
    if "mean difference" in lowered or "difference in means" in lowered:
        return "mean_difference"
    if "risk difference" in lowered:
        return "risk_difference"
    if "rate difference" in lowered:
        return "rate_difference"
    return None


def extract_ctgov_effect_candidates(study_payload: dict[str, Any]) -> list[ExtractedEffectCandidate]:
    study = _coerce_study_payload(study_payload)
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {}) if isinstance(protocol, dict) else {}
    study_id = _string_value(identification.get("nctId"), "ctgov-study")

    results = study.get("resultsSection", {})
    if not isinstance(results, dict):
        return []

    group_titles = _collect_ctgov_group_titles(study)
    candidates: list[ExtractedEffectCandidate] = []

    outcome_module = results.get("outcomeMeasuresModule", {})
    outcome_measures = outcome_module.get("outcomeMeasures", []) if isinstance(outcome_module, dict) else []
    if not isinstance(outcome_measures, list):
        return candidates

    for outcome in outcome_measures:
        if not isinstance(outcome, dict):
            continue
        outcome_title = _string_value(outcome.get("title"), outcome.get("measure"), outcome.get("description"))
        local_group_titles = {
            **group_titles,
            **_group_titles_from_group_list(outcome.get("groups")),
        }
        analyses = outcome.get("analyses", [])
        if isinstance(analyses, list):
            for analysis in analyses:
                if not isinstance(analysis, dict):
                    continue
                statistic_text = " ".join(
                    part
                    for part in (
                        _string_value(analysis.get("statistic")),
                        _string_value(analysis.get("paramType")),
                        _string_value(analysis.get("method")),
                        _string_value(analysis.get("otherAnalysisDescription")),
                    )
                    if part
                )
                metric = _infer_ctgov_metric(statistic_text)
                if not metric:
                    continue
                estimate = _parse_float(analysis.get("estimate"))
                ci_low = _parse_float(analysis.get("ciLowerLimit"))
                ci_high = _parse_float(analysis.get("ciUpperLimit"))
                if estimate is None or ci_low is None or ci_high is None:
                    continue

                group_ids = _extract_group_ids(analysis.get("groups"))
                if not group_ids:
                    group_ids = _extract_group_ids(analysis.get("groupIds"))
                candidates.append(
                    candidate_from_estimate_and_ci(
                        source_record_id=study_id,
                        source="ctgov",
                        metric=metric,
                        estimate=estimate,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        outcome=outcome_title,
                        comparator=_comparator_from_group_ids(group_ids, local_group_titles),
                        snippet=" ".join(
                            part
                            for part in (
                                outcome_title,
                                statistic_text,
                                f"{estimate} (95% CI {ci_low} to {ci_high})",
                            )
                            if part
                        ),
                        extraction_method="ctgov_analysis_ci",
                        metadata={
                            "group_ids": group_ids,
                            "method": _string_value(analysis.get("method")),
                            "p_value": _parse_float(analysis.get("pValue")),
                        },
                    )
                )

        unit = _string_value(outcome.get("unitOfMeasure")).lower()
        denoms = _extract_denominators(outcome)
        if not denoms:
            continue

        dispersion_type = _string_value(outcome.get("dispersionType")).lower()
        param_type = _string_value(outcome.get("paramType")).lower()
        for set_label, measurements in _iter_measurement_sets(outcome):
            if len(measurements) != 2:
                continue

            values: list[tuple[str, float, float, float | None]] = []
            for measurement in measurements:
                if not isinstance(measurement, dict):
                    values = []
                    break
                group_id = _string_value(measurement.get("groupId"), measurement.get("id"))
                value = _parse_float(measurement.get("value"))
                spread = _parse_float(measurement.get("spread"))
                denom = denoms.get(group_id)
                if not group_id or value is None or denom is None:
                    values = []
                    break
                values.append((group_id, value, denom, spread))

            if len(values) != 2:
                continue

            raw_group_a, raw_value_a, raw_denom_a, raw_spread_a = values[0]
            raw_group_b, raw_value_b, raw_denom_b, raw_spread_b = values[1]
            group_a, group_b = _ordered_pair(raw_group_a, raw_group_b, local_group_titles)
            by_group = {
                raw_group_a: (raw_value_a, raw_denom_a, raw_spread_a),
                raw_group_b: (raw_value_b, raw_denom_b, raw_spread_b),
            }
            value_a, denom_a, spread_a = by_group[group_a]
            value_b, denom_b, spread_b = by_group[group_b]
            comparator = _comparator_from_group_ids([group_a, group_b], local_group_titles)
            outcome_label = " / ".join(part for part in (outcome_title, set_label) if part)

            if any(token in unit for token in ("participant", "participants", "subject", "event")) and denom_a > value_a and denom_b > value_b:
                effect_size, variance = log_odds_ratio(
                    int(round(value_a)),
                    int(round(denom_a - value_a)),
                    int(round(value_b)),
                    int(round(denom_b - value_b)),
                )
                candidates.append(
                    ExtractedEffectCandidate(
                        source_record_id=study_id,
                        source="ctgov",
                        metric="or",
                        effect_size=effect_size,
                        variance=variance,
                        effect_scale="log",
                        outcome=outcome_label,
                        comparator=comparator,
                        snippet=f"{local_group_titles.get(group_a, group_a)} {value_a}/{denom_a} vs {local_group_titles.get(group_b, group_b)} {value_b}/{denom_b}",
                        extraction_method="ctgov_measurement_counts",
                        metadata={
                            "group_ids": [group_a, group_b],
                            "group_values": {
                                group_a: {"events": value_a, "denominator": denom_a},
                                group_b: {"events": value_b, "denominator": denom_b},
                            },
                            "unit_of_measure": unit,
                        },
                    )
                )
                continue

            if param_type == "mean" and dispersion_type == "standard deviation" and spread_a is not None and spread_b is not None:
                variance = ((spread_a ** 2) / denom_a) + ((spread_b ** 2) / denom_b)
                if variance <= 0:
                    continue
                candidates.append(
                    ExtractedEffectCandidate(
                        source_record_id=study_id,
                        source="ctgov",
                        metric="mean_difference",
                        effect_size=value_a - value_b,
                        variance=variance,
                        effect_scale="identity",
                        outcome=outcome_label,
                        comparator=comparator,
                        estimate=value_a - value_b,
                        snippet=f"{local_group_titles.get(group_a, group_a)} mean {value_a} (SD {spread_a}, n={denom_a}) vs {local_group_titles.get(group_b, group_b)} mean {value_b} (SD {spread_b}, n={denom_b})",
                        extraction_method="ctgov_measurement_means",
                        metadata={
                            "group_ids": [group_a, group_b],
                            "group_values": {
                                group_a: {"mean": value_a, "sd": spread_a, "denominator": denom_a},
                                group_b: {"mean": value_b, "sd": spread_b, "denominator": denom_b},
                            },
                            "unit_of_measure": unit,
                        },
                    )
                )

    return candidates
