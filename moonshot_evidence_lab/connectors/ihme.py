from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class IHMEDatasetRecord:
    slug: str
    title: str
    source_url: str
    license: str
    local_path: str
    notes: str

    def to_dict(self) -> dict[str, str]:
        return {
            "slug": self.slug,
            "title": self.title,
            "source_url": self.source_url,
            "license": self.license,
            "local_path": self.local_path,
            "notes": self.notes,
        }


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_template_path() -> Path:
    return project_root() / "data" / "registries" / "ihme_catalog.template.csv"


def template_records() -> list[IHMEDatasetRecord]:
    return [
        IHMEDatasetRecord(
            slug="gbd-results-2021",
            title="Global Burden of Disease Results 2021",
            source_url="https://vizhub.healthdata.org/gbd-results/",
            license="IHME non-commercial agreement",
            local_path="data/raw/ihme/gbd-results-2021/",
            notes="Export CSV files after sign-in and record filters used.",
        ),
        IHMEDatasetRecord(
            slug="air-pollution-exposure-2021",
            title="GBD 2021 Air Pollution Exposure Estimates",
            source_url="https://ghdx.healthdata.org/ihme_data",
            license="IHME non-commercial agreement",
            local_path="data/raw/ihme/air-pollution-exposure-2021/",
            notes="Keep release notes and any codebooks next to the raw files.",
        ),
    ]


def write_template_catalog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = template_records()
    fieldnames = list(records[0].to_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def load_catalog(path: Path) -> list[IHMEDatasetRecord]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            IHMEDatasetRecord(
                slug=row["slug"],
                title=row["title"],
                source_url=row["source_url"],
                license=row["license"],
                local_path=row["local_path"],
                notes=row["notes"],
            )
            for row in reader
        ]


def download_instructions() -> str:
    return (
        "1. Register or sign in to IHME/GHDx.\n"
        "2. Export the dataset or download the files you are licensed to use.\n"
        "3. Store raw files under data/raw/ihme/<slug>/.\n"
        "4. Record provenance and notes in the catalog CSV.\n"
        "5. Normalize the downloaded tables before modeling."
    )
