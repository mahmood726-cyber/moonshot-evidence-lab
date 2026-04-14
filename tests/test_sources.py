from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from moonshot_evidence_lab.connectors.clinicaltrials import ClinicalTrialsGovClient
from moonshot_evidence_lab.connectors.ihme import load_catalog, write_template_catalog
from moonshot_evidence_lab.connectors.pubmed import PubMedClient
from moonshot_evidence_lab.connectors.who import WHOClient


class SourceConnectorTests(unittest.TestCase):
    def test_who_url_builder(self) -> None:
        client = WHOClient()
        url = client.build_indicator_url("WHOSIS_000001", top=5, filter_expr="Dim1 eq 'MLE'")
        self.assertIn("WHOSIS_000001", url)
        self.assertIn("$top=5", url)
        self.assertIn("$filter=Dim1%20eq%20'MLE'", url)

    def test_pubmed_url_builder(self) -> None:
        client = PubMedClient()
        url = client.build_esearch_url("stroke prevention", retmax=5)
        self.assertIn("db=pubmed", url)
        self.assertIn("term=stroke+prevention", url)
        self.assertIn("retmax=5", url)

    def test_ctgov_url_builder(self) -> None:
        client = ClinicalTrialsGovClient()
        url = client.build_studies_url(term="aspirin", condition="stroke", page_size=10)
        self.assertIn("query.term=aspirin", url)
        self.assertIn("query.cond=stroke", url)
        self.assertIn("pageSize=10", url)

    def test_ihme_template_round_trip(self) -> None:
        temp_root = Path(__file__).resolve().parents[1] / ".tmp_test_sources"
        temp_root.mkdir(exist_ok=True)
        try:
            path = temp_root / "ihme_catalog.csv"
            write_template_catalog(path)
            catalog = load_catalog(path)
            self.assertGreaterEqual(len(catalog), 2)
            self.assertEqual(catalog[0].slug, "gbd-results-2021")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
