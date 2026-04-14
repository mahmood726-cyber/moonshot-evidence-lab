from __future__ import annotations

import unittest

from moonshot_evidence_lab.benchmarks import run_literature_screening_benchmark
from moonshot_evidence_lab.connectors.clinicaltrials import ClinicalTrialsGovClient
from moonshot_evidence_lab.connectors.pubmed import PubMedArticle
from moonshot_evidence_lab.literature import clinical_trial_to_record, pubmed_article_to_record
from moonshot_evidence_lab.models import EvidenceRecord


class LiteratureTests(unittest.TestCase):
    def test_pubmed_article_is_normalized(self) -> None:
        article = PubMedArticle(
            pmid="12345",
            title="Systematic review of stroke prevention",
            pubdate="2024 May",
            journal="Example Journal",
            article_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
            metadata={"pubtype": ["Systematic Review"]},
        )
        record = pubmed_article_to_record(article)
        self.assertIsInstance(record, EvidenceRecord)
        self.assertEqual(record.source, "systematic_review")
        self.assertEqual(record.year, 2024)

    def test_ctgov_trial_is_normalized(self) -> None:
        client = ClinicalTrialsGovClient()
        payload = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Aspirin for stroke prevention",
                            "officialTitle": "A Randomized Trial of Aspirin",
                        },
                        "statusModule": {
                            "overallStatus": "COMPLETED",
                            "primaryCompletionDateStruct": {"date": "2024-10-01"},
                        },
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE3"],
                            "enrollmentInfo": {"count": 420},
                        },
                        "conditionsModule": {"conditions": ["Stroke"]},
                        "armsInterventionsModule": {"interventions": [{"name": "Aspirin"}]},
                    },
                    "hasResults": True,
                }
            ]
        }
        records = client.normalize_studies(payload)
        self.assertEqual(len(records), 1)
        record = clinical_trial_to_record(records[0])
        self.assertEqual(record.record_id, "ctgov:NCT12345678")
        self.assertEqual(record.year, 2024)
        self.assertTrue(record.metadata["has_results"])

    def test_literature_benchmark_scores_perfectly_for_default_policy(self) -> None:
        result = run_literature_screening_benchmark()
        self.assertEqual(result["records"], 8)
        self.assertEqual(result["true_positive"], 4)
        self.assertEqual(result["false_positive"], 0)
        self.assertEqual(result["false_negative"], 0)
        self.assertEqual(result["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
