from __future__ import annotations

import unittest

from moonshot_evidence_lab.connectors.pubmed import PubMedClient
from moonshot_evidence_lab.extraction import extract_ctgov_effect_candidates, extract_effect_candidates_from_text


PUBMED_EFETCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Trial of aspirin for stroke prevention</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
              <Month>Jul</Month>
              <Day>12</Day>
            </PubDate>
          </JournalIssue>
          <Title>Example Journal</Title>
        </Journal>
        <Abstract>
          <AbstractText Label="Results">The odds ratio was 0.75 (95% CI 0.60 to 0.90).</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <Initials>AB</Initials>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/example</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


class ExtractionTests(unittest.TestCase):
    def test_pubmed_xml_parser_extracts_abstract(self) -> None:
        client = PubMedClient()
        articles = client.parse_efetch_xml(PUBMED_EFETCH_XML)
        self.assertEqual(len(articles), 1)
        article = articles[0]
        self.assertEqual(article.pmid, "12345678")
        self.assertEqual(article.pubdate, "2024 Jul 12")
        self.assertIn("Results:", article.abstract)

    def test_text_ratio_extraction(self) -> None:
        text = "At 12 months the hazard ratio was 0.84 (95% CI 0.72 to 0.98)."
        candidates = extract_effect_candidates_from_text(
            text,
            source_record_id="pubmed:1",
            source="pubmed",
            outcome="stroke",
        )
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.metric, "hr")
        self.assertLess(candidate.effect_size, 0.0)
        self.assertGreater(candidate.variance, 0.0)

    def test_text_binary_count_extraction(self) -> None:
        text = "Primary events occurred in 12/100 vs 20/100 participants."
        candidates = extract_effect_candidates_from_text(
            text,
            source_record_id="pubmed:2",
            source="pubmed",
            outcome="mortality",
        )
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.metric, "or")
        self.assertEqual(candidate.extraction_method, "text_binary_counts")
        self.assertGreater(candidate.variance, 0.0)

    def test_ctgov_results_analysis_extraction(self) -> None:
        payload = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT00000000",
                },
                "armsInterventionsModule": {
                    "armGroups": [
                        {"id": "G1", "label": "Aspirin"},
                        {"id": "G2", "label": "Placebo"},
                    ]
                },
            },
            "resultsSection": {
                "participantFlowModule": {
                    "groups": [
                        {"id": "G1", "title": "Aspirin"},
                        {"id": "G2", "title": "Placebo"},
                    ]
                },
                "outcomeMeasuresModule": {
                    "outcomeMeasures": [
                        {
                            "title": "Stroke at 12 months",
                            "analyses": [
                                {
                                    "statistic": "Odds Ratio",
                                    "estimate": "0.75",
                                    "ciLowerLimit": "0.60",
                                    "ciUpperLimit": "0.94",
                                    "groups": [{"groupId": "G1"}, {"groupId": "G2"}],
                                }
                            ],
                        }
                    ]
                },
            },
        }
        candidates = extract_ctgov_effect_candidates(payload)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.source_record_id, "NCT00000000")
        self.assertEqual(candidate.metric, "or")
        self.assertIn("Aspirin", candidate.comparator)

    def test_ctgov_measurement_extraction_from_counts_and_means(self) -> None:
        payload = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT00000001"},
            },
            "resultsSection": {
                "outcomeMeasuresModule": {
                    "outcomeMeasures": [
                        {
                            "title": "Recurrent stroke",
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
                                        {"groupId": "OG000", "value": "83"},
                                        {"groupId": "OG001", "value": "81"},
                                    ],
                                }
                            ],
                            "classes": [
                                {
                                    "categories": [
                                        {
                                            "measurements": [
                                                {"groupId": "OG000", "value": "4"},
                                                {"groupId": "OG001", "value": "2"},
                                            ]
                                        }
                                    ]
                                }
                            ],
                        },
                        {
                            "title": "Change in pulsatility index",
                            "paramType": "MEAN",
                            "dispersionType": "Standard Deviation",
                            "unitOfMeasure": "ratio",
                            "groups": [
                                {"id": "OG000", "title": "Placebo"},
                                {"id": "OG001", "title": "Aspirin"},
                            ],
                            "denoms": [
                                {
                                    "units": "Participants",
                                    "counts": [
                                        {"groupId": "OG000", "value": "83"},
                                        {"groupId": "OG001", "value": "81"},
                                    ],
                                }
                            ],
                            "classes": [
                                {
                                    "title": "Right MCA",
                                    "categories": [
                                        {
                                            "measurements": [
                                                {"groupId": "OG000", "value": "0.09", "spread": "0.13"},
                                                {"groupId": "OG001", "value": "0.14", "spread": "0.13"},
                                            ]
                                        }
                                    ]
                                }
                            ],
                        },
                    ]
                }
            },
        }
        candidates = extract_ctgov_effect_candidates(payload)
        metrics = {candidate.metric for candidate in candidates}
        methods = {candidate.extraction_method for candidate in candidates}
        self.assertIn("or", metrics)
        self.assertIn("mean_difference", metrics)
        self.assertIn("ctgov_measurement_counts", methods)
        self.assertIn("ctgov_measurement_means", methods)


if __name__ == "__main__":
    unittest.main()
