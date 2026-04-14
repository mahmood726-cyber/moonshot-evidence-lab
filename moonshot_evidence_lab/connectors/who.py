from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_WHO_GHO_BASE_URL = os.getenv("WHO_GHO_BASE_URL", "https://ghoapi.azureedge.net/api")
WHO_TRANSITION_NOTE = (
    "WHO has been transitioning from the legacy GHO OData endpoint to newer Data Hub infrastructure. "
    "Keep WHO_GHO_BASE_URL configurable so the client can be repointed without code changes."
)


def _query_string(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in params.items():
        if value in (None, ""):
            continue
        encoded_key = quote(str(key), safe="$")
        encoded_value = quote(str(value), safe="(),'/:")
        parts.append(f"{encoded_key}={encoded_value}")
    return "&".join(parts)


@dataclass(slots=True)
class WHOClient:
    base_url: str = DEFAULT_WHO_GHO_BASE_URL
    timeout_seconds: int = 30

    def build_indicator_url(
        self,
        indicator_code: str,
        *,
        top: int | None = 100,
        filter_expr: str | None = None,
        select_fields: list[str] | None = None,
    ) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/{indicator_code}"
        params: dict[str, Any] = {}
        if top is not None:
            params["$top"] = top
        if filter_expr:
            params["$filter"] = filter_expr
        if select_fields:
            params["$select"] = ",".join(select_fields)

        query = _query_string(params)
        return endpoint if not query else f"{endpoint}?{query}"

    def build_indicator_catalog_url(self, *, contains: str | None = None) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/Indicator"
        if not contains:
            return endpoint
        filter_expr = f"contains(IndicatorName,'{contains}')"
        return f"{endpoint}?{_query_string({'$filter': filter_expr})}"

    def fetch_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object response")
        return payload

    def fetch_indicator_rows(
        self,
        indicator_code: str,
        *,
        top: int | None = 100,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = self.fetch_json(
            self.build_indicator_url(indicator_code, top=top, filter_expr=filter_expr)
        )
        if "value" in payload and isinstance(payload["value"], list):
            return [row for row in payload["value"] if isinstance(row, dict)]
        d_section = payload.get("d")
        if isinstance(d_section, dict) and isinstance(d_section.get("results"), list):
            return [row for row in d_section["results"] if isinstance(row, dict)]
        raise ValueError("WHO response did not contain recognizable OData rows")
