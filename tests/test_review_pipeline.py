from __future__ import annotations

import unittest

from moonshot_evidence_lab.connectors.clinicaltrials import ClinicalTrialRecord
from moonshot_evidence_lab.connectors.pubmed import PubMedArticle
from moonshot_evidence_lab.review_pipeline import canonical_outcome_tag, run_review_pipeline


class FakePubMedClient:
    def fetch_articles(self, pubmed_ids: list[str] | tuple[str, ...]) -> list[PubMedArticle]:
        return [
            PubMedArticle(
                pmid=str(pubmed_ids[0]),
                title="Trial of aspirin for recurrent stroke prevention",
                pubdate="2024 Jan",
                journal="Example Journal",
                article_url=f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_ids[0]}/",
                abstract="The odds ratio was 0.80 (95% CI 0.70 to 0.92) for recurrent stroke.",
                metadata={"source": "pubmed", "pubtype": ["Journal Article"]},
            )
        ]

    def search_and_fetch_abstracts(self, term: str, *, retmax: int = 20, sort: str = "relevance") -> list[PubMedArticle]:
        return self.fetch_articles(["111"])


class FakeClinicalTrialsGovClient:
    def search_studies(
        self,
        *,
        term: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        page_size: int = 20,
    ) -> list[ClinicalTrialRecord]:
        return [
            ClinicalTrialRecord(
                nct_id="NCTFAKE1",
                brief_title="Aspirin versus placebo for recurrent stroke",
                official_title="Aspirin versus placebo for recurrent stroke",
                overall_status="COMPLETED",
                study_type="INTERVENTIONAL",
                phases=["PHASE3"],
                conditions=["Stroke"],
                interventions=["Aspirin", "Placebo"],
                enrollment=200,
                has_results=True,
                primary_completion_date="2024-06-01",
                study_url="https://clinicaltrials.gov/study/NCTFAKE1",
                metadata={"source": "ctgov", "study_design": "interventional", "has_results": True},
            )
        ]

    def fetch_study(self, nct_id: str) -> dict:
        return {
            "protocolSection": {
                "identificationModule": {"nctId": nct_id},
            },
            "resultsSection": {
                "outcomeMeasuresModule": {
                    "outcomeMeasures": [
                        {
                            "title": "Number of Patients With First Recurrent Stroke of Any Type",
                            "paramType": "NUMBER",
                            "unitOfMeasure": "participants",
                            "groups": [
                                {"id": "OG000", "title": "Placebo"},
                                {"id": "OG001", "title": "Aspirin"},
                            ],
                            "denoms": [
                                {
                                    "units": "Participants",
                                    "counts": [
                                        {"groupId": "OG000", "value": "100"},
                                        {"groupId": "OG001", "value": "100"},
                                    ],
                                }
                            ],
                            "classes": [
                                {
                                    "categories": [
                                        {
                                            "measurements": [
                                                {"groupId": "OG000", "value": "20"},
                                                {"groupId": "OG001", "value": "12"},
                                            ]
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                }
            },
            "hasResults": True,
        }

    def normalize_studies(self, payload: dict) -> list[ClinicalTrialRecord]:
        nct_id = payload["protocolSection"]["identificationModule"]["nctId"]
        return [
            ClinicalTrialRecord(
                nct_id=nct_id,
                brief_title="Aspirin versus placebo for recurrent stroke",
                official_title="Aspirin versus placebo for recurrent stroke",
                overall_status="COMPLETED",
                study_type="INTERVENTIONAL",
                phases=["PHASE3"],
                conditions=["Stroke"],
                interventions=["Aspirin", "Placebo"],
                enrollment=200,
                has_results=True,
                primary_completion_date="2024-06-01",
                study_url=f"https://clinicaltrials.gov/study/{nct_id}",
                metadata={"source": "ctgov", "study_design": "interventional", "has_results": True},
            )
        ]


class FailingPubMedClient:
    def search_and_fetch_abstracts(self, term: str, *, retmax: int = 20, sort: str = "relevance") -> list[PubMedArticle]:
        raise RuntimeError("pubmed unavailable")


class ReviewPipelineTests(unittest.TestCase):
    def test_canonical_outcome_tag(self) -> None:
        self.assertEqual(canonical_outcome_tag("Hemorrhagic stroke"), "hemorrhagic_stroke")
        self.assertEqual(canonical_outcome_tag("All-cause mortality"), "all_cause_mortality")
        self.assertEqual(canonical_outcome_tag("Changes in pulsatility index"), "pulsatility_index")

    def test_review_pipeline_pools_across_sources(self) -> None:
        result = run_review_pipeline(
            term="aspirin stroke",
            condition="stroke",
            pubmed_retmax=3,
            ctgov_page_size=3,
            pubmed_client=FakePubMedClient(),
            ctgov_client=FakeClinicalTrialsGovClient(),
        )
        self.assertEqual(result["records"]["pubmed_found"], 1)
        self.assertEqual(result["records"]["ctgov_found"], 1)
        self.assertEqual(result["candidates"]["deduplicated_effects"], 2)
        self.assertEqual(len(result["pooled_analyses"]), 1)
        pooled = result["pooled_analyses"][0]
        self.assertEqual(pooled["outcome_tag"], "stroke")
        self.assertEqual(pooled["metric"], "or")
        self.assertEqual(pooled["k"], 2)

    def test_review_pipeline_records_source_errors_without_crashing(self) -> None:
        result = run_review_pipeline(
            term="aspirin stroke",
            condition="stroke",
            pubmed_retmax=3,
            ctgov_page_size=3,
            pubmed_client=FailingPubMedClient(),
            ctgov_client=FakeClinicalTrialsGovClient(),
        )
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["source"], "pubmed")
        self.assertGreaterEqual(result["candidates"]["ctgov"], 1)

    def test_review_pipeline_accepts_seeded_ids(self) -> None:
        result = run_review_pipeline(
            term="",
            pubmed_pmids=["31237833"],
            ctgov_nct_ids=["NCT00741286"],
            include_secondary_evidence=True,
            pubmed_client=FakePubMedClient(),
            ctgov_client=FakeClinicalTrialsGovClient(),
        )
        self.assertEqual(result["query"]["pubmed_pmids"], ["31237833"])
        self.assertEqual(result["query"]["ctgov_nct_ids"], ["NCT00741286"])
        self.assertEqual(len(result["pooled_analyses"]), 1)


if __name__ == "__main__":
    unittest.main()
