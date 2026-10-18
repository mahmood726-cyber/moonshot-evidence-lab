"""Pagination transport fixtures, not research records."""
from urllib.parse import parse_qs, urlparse

import unittest
from unittest.mock import patch

from moonshot_evidence_lab.connectors.clinicaltrials import ClinicalTrialsGovClient


def study(label):
    return {"protocolSection": {"identificationModule": {"briefTitle": label}}}


def install_pages(test_case, pages):
    urls = []
    def fetch(self, url):
        urls.append(url)
        assert len(urls) <= len(pages), "pagination did not terminate"
        page = pages[len(urls) - 1]
        if isinstance(page, Exception):
            raise page
        return page
    patcher = patch.object(ClinicalTrialsGovClient, "fetch_json", fetch)
    patcher.start()
    test_case.addCleanup(patcher.stop)
    return urls


class PaginationTests(unittest.TestCase):
    def test_search_collects_every_page_and_preserves_filters(self):
        token = "opaque +/=& token"
        urls = install_pages(self, [
            {"studies": [study("fixture first")], "nextPageToken": token},
            {"studies": [study("fixture second")]},
        ])
        records = ClinicalTrialsGovClient().search_studies(term="aspirin", condition="stroke", page_size=1)
        assert [r.brief_title for r in records] == ["fixture first", "fixture second"]
        query = parse_qs(urlparse(urls[1]).query)
        assert query["pageToken"] == [token]
        assert query["query.term"] == ["aspirin"]
        assert query["query.cond"] == ["stroke"]
        assert query["pageSize"] == ["1"]


    def test_invalid_pages_cannot_become_empty_searches(self):
        for page in [[], {}, {"error": "failure"}, {"studies": None},
                     {"studies": {}}, {"studies": [None]},
                     {"studies": [], "nextPageToken": 4},
                     {"studies": [], "nextPageToken": ""}]:
            with self.subTest(page=page):
                install_pages(self, [page])
                with self.assertRaises(ValueError):
                    ClinicalTrialsGovClient().search_studies(term="aspirin")


    def test_repeated_token_fails_closed(self):
        urls = install_pages(self, [{"studies": [], "nextPageToken": "cycle"}] * 2)
        with self.assertRaisesRegex(ValueError, "token"):
            ClinicalTrialsGovClient().search_studies(term="aspirin")
        assert len(urls) == 2


    def test_later_request_failure_does_not_return_partial_result(self):
        install_pages(self, [
            {"studies": [study("fixture first")], "nextPageToken": "next"},
            RuntimeError("transport fixture failure"),
        ])
        with self.assertRaisesRegex(RuntimeError, "transport fixture"):
            ClinicalTrialsGovClient().search_studies(term="aspirin")


    def test_empty_final_page_is_valid(self):
        install_pages(self, [{"studies": []}])
        assert ClinicalTrialsGovClient().search_studies(term="aspirin") == []
