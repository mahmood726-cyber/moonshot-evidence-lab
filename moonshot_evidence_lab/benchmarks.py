from __future__ import annotations

from pathlib import Path

from .literature import load_screening_fixture
from .program_surface import DEFAULT_POLICY, include_record


def literature_benchmark_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "literature_screening_fixture.json"


def run_literature_screening_benchmark() -> dict[str, float | int]:
    records = load_screening_fixture(literature_benchmark_path())
    gold_positive = {
        record.record_id
        for record in records
        if bool(record.metadata.get("gold_include", False))
    }
    predicted_positive = {
        record.record_id
        for record in records
        if include_record(record.metadata, policy=DEFAULT_POLICY)
    }

    true_positive = len(gold_positive & predicted_positive)
    false_positive = len(predicted_positive - gold_positive)
    false_negative = len(gold_positive - predicted_positive)

    precision = true_positive / len(predicted_positive) if predicted_positive else 0.0
    recall = true_positive / len(gold_positive) if gold_positive else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "records": len(records),
        "gold_positive": len(gold_positive),
        "predicted_positive": len(predicted_positive),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
