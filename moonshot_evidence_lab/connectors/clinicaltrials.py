from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_CLINICALTRIALS_GOV_API_BASE_URL = os.getenv(
    "CTGOV_API_BASE_URL",
    "https://clinicaltrials.gov/api/v2",
)


@dataclass(slots=True)
class ClinicalTrialRecord:
    nct_id: str
    brief_title: str
    official_title: str = ""
    overall_status: str = ""
    study_type: str = ""
    phases: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    interventions: list[str] = field(default_factory=list)
    enrollment: int | None = None
    has_results: bool = False
    primary_completion_date: str = ""
    study_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _query_string(params: dict[str, Any]) -> str:
    return urlencode({key: value for key, value in params.items() if value not in ("", None)}, doseq=True)


@dataclass(slots=True)
class ClinicalTrialsGovClient:
    base_url: str = DEFAULT_CLINICALTRIALS_GOV_API_BASE_URL
    timeout_seconds: int = 30

    def build_studies_url(
        self,
        *,
        term: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        page_size: int = 20,
        count_total: bool = True,
    ) -> str:
        params: dict[str, Any] = {"pageSize": page_size}
        if count_total:
            params["countTotal"] = "true"
        if term:
            params["query.term"] = term
        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        return f"{self.base_url.rstrip('/')}/studies?{_query_string(params)}"

    def build_study_url(self, nct_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/studies/{nct_id}"

    def fetch_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object response")
        return payload

    def fetch_study(self, nct_id: str) -> dict[str, Any]:
        return self.fetch_json(self.build_study_url(nct_id))

    def _study_objects(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        studies = payload.get("studies")
        if isinstance(studies, list):
            return [study for study in studies if isinstance(study, dict)]
        if isinstance(payload.get("study"), dict):
            return [payload["study"]]
        if payload.get("protocolSection") or payload.get("resultsSection"):
            return [payload]
        return []

    def normalize_studies(self, payload: dict[str, Any]) -> list[ClinicalTrialRecord]:
        records: list[ClinicalTrialRecord] = []
        for study in self._study_objects(payload):
            protocol = study.get("protocolSection", {})
            if not isinstance(protocol, dict):
                continue

            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            arms_module = protocol.get("armsInterventionsModule", {})
            if not isinstance(identification, dict):
                identification = {}
            if not isinstance(status, dict):
                status = {}
            if not isinstance(design, dict):
                design = {}
            if not isinstance(conditions_module, dict):
                conditions_module = {}
            if not isinstance(arms_module, dict):
                arms_module = {}

            interventions: list[str] = []
            for item in arms_module.get("interventions", []):
                if isinstance(item, dict) and item.get("name"):
                    interventions.append(str(item["name"]).strip())

            enrollment_value = None
            enrollment_info = design.get("enrollmentInfo", {})
            if isinstance(enrollment_info, dict) and enrollment_info.get("count") not in (None, ""):
                try:
                    enrollment_value = int(enrollment_info["count"])
                except (TypeError, ValueError):
                    enrollment_value = None

            phases = design.get("phases", [])
            conditions = conditions_module.get("conditions", [])
            has_results = bool(study.get("hasResults", False) or study.get("resultsSection"))
            nct_id = str(identification.get("nctId", "")).strip()

            records.append(
                ClinicalTrialRecord(
                    nct_id=nct_id,
                    brief_title=str(identification.get("briefTitle", "")).strip(),
                    official_title=str(identification.get("officialTitle", "")).strip(),
                    overall_status=str(status.get("overallStatus", "")).strip(),
                    study_type=str(design.get("studyType", "")).strip(),
                    phases=[str(phase).strip() for phase in phases if str(phase).strip()],
                    conditions=[str(condition).strip() for condition in conditions if str(condition).strip()],
                    interventions=interventions,
                    enrollment=enrollment_value,
                    has_results=has_results,
                    primary_completion_date=str(status.get("primaryCompletionDateStruct", {}).get("date", "")).strip(),
                    study_url=f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
                    metadata={
                        "source": "ctgov",
                        "study_design": str(design.get("studyType", "")).strip().lower(),
                        "has_results": has_results,
                    },
                )
            )
        return records

    def search_studies(
        self,
        *,
        term: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        page_size: int = 20,
    ) -> list[ClinicalTrialRecord]:
        payload = self.fetch_json(
            self.build_studies_url(
                term=term,
                condition=condition,
                intervention=intervention,
                page_size=page_size,
            )
        )
        return self.normalize_studies(payload)
