from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import StudyEffect
from .program_surface import DEFAULT_POLICY, include_record


def effect_from_mapping(data: Mapping[str, Any]) -> StudyEffect:
    return StudyEffect(
        study_id=str(data["study_id"]),
        label=str(data.get("label", data["study_id"])),
        effect_size=float(data["effect_size"]),
        variance=float(data["variance"]),
        outcome=str(data.get("outcome", "")),
        comparator=str(data.get("comparator", "")),
        metadata=dict(data.get("metadata", {})),
    )


def load_effects(path: str | Path) -> list[StudyEffect]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("expected a list of effect records")
    return [effect_from_mapping(item) for item in payload]


def filter_effects(
    effects: list[StudyEffect],
    *,
    policy=DEFAULT_POLICY,
) -> list[StudyEffect]:
    return [effect for effect in effects if include_record(effect.metadata, policy=policy)]
