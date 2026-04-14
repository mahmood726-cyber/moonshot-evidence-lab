from __future__ import annotations

import json
import re
from pathlib import Path

from .connectors.clinicaltrials import ClinicalTrialRecord
from .connectors.pubmed import PubMedArticle
from .models import EvidenceRecord
from .program_surface import DEFAULT_POLICY, include_record

YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def extract_year(value: str) -> int | None:
    match = YEAR_PATTERN.search(value)
    if not match:
        return None
    return int(match.group(0))


def pubmed_article_to_record(article: PubMedArticle) -> EvidenceRecord:
    publication_types = article.metadata.get("pubtype", [])
    normalized_types = [str(item).strip().lower() for item in publication_types if str(item).strip()]
    source = "systematic_review" if any("review" in item for item in normalized_types) else "pubmed"

    return EvidenceRecord(
        record_id=f"pubmed:{article.pmid}",
        source=source,
        title=article.title,
        year=extract_year(article.pubdate),
        abstract=article.abstract,
        url=article.article_url,
        metadata={
            "source": source,
            "year": extract_year(article.pubdate),
            "publication_types": normalized_types,
            "journal": article.journal,
            "doi": article.doi,
            "authors": article.authors,
            "sample_size": 0,
            "missingness": 0.0,
        },
    )


def clinical_trial_to_record(trial: ClinicalTrialRecord) -> EvidenceRecord:
    title = trial.brief_title or trial.official_title or trial.nct_id
    year = extract_year(trial.primary_completion_date)
    study_design = "trial" if trial.study_type.lower() == "interventional" else "registry"

    return EvidenceRecord(
        record_id=f"ctgov:{trial.nct_id}",
        source="ctgov",
        title=title,
        year=year,
        url=trial.study_url,
        metadata={
            "source": "ctgov",
            "year": year,
            "study_design": study_design,
            "trial_status": trial.overall_status,
            "has_results": trial.has_results,
            "sample_size": trial.enrollment or 0,
            "missingness": 0.0,
            "conditions": trial.conditions,
            "interventions": trial.interventions,
            "phases": trial.phases,
        },
    )


def filter_literature_records(
    records: list[EvidenceRecord],
    *,
    policy=DEFAULT_POLICY,
) -> list[EvidenceRecord]:
    return [record for record in records if include_record(record.metadata, policy=policy)]


def load_screening_fixture(path: str | Path) -> list[EvidenceRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("expected a list of literature benchmark records")

    records: list[EvidenceRecord] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        year_raw = item.get("year")
        year = int(year_raw) if year_raw not in (None, "") else None
        metadata = dict(item.get("metadata", {}))
        if year is not None:
            metadata.setdefault("year", year)
        records.append(
            EvidenceRecord(
                record_id=str(item["record_id"]),
                source=str(item["source"]),
                title=str(item["title"]),
                year=year,
                abstract=str(item.get("abstract", "")),
                url=str(item.get("url", "")),
                metadata=metadata,
            )
        )
    return records
