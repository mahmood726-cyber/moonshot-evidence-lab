from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from defusedxml import ElementTree as ET
DEFAULT_NCBI_EUTILS_BASE_URL = os.getenv(
    "NCBI_EUTILS_BASE_URL",
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
)


@dataclass(slots=True)
class PubMedArticle:
    pmid: str
    title: str
    pubdate: str = ""
    journal: str = ""
    doi: str = ""
    authors: list[str] = field(default_factory=list)
    article_url: str = ""
    abstract: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _query_string(params: dict[str, Any]) -> str:
    return urlencode({key: value for key, value in params.items() if value not in ("", None)})


@dataclass(slots=True)
class PubMedClient:
    base_url: str = DEFAULT_NCBI_EUTILS_BASE_URL
    tool: str = os.getenv("NCBI_TOOL", "moonshot-evidence-lab")
    email: str = os.getenv("NCBI_EMAIL", "")
    api_key: str = os.getenv("NCBI_API_KEY", "")
    timeout_seconds: int = 30

    def _default_params(self, *, include_credentials: bool) -> dict[str, Any]:
        params: dict[str, Any] = {"retmode": "json", "tool": self.tool}
        if include_credentials and self.email:
            params["email"] = self.email
        if include_credentials and self.api_key:
            params["api_key"] = self.api_key
        return params

    def build_esearch_url(
        self,
        term: str,
        *,
        retmax: int = 20,
        sort: str = "relevance",
        use_history: bool = False,
        include_credentials: bool = False,
    ) -> str:
        params = self._default_params(include_credentials=include_credentials)
        params.update(
            {
                "db": "pubmed",
                "term": term,
                "retmax": retmax,
                "sort": sort,
            }
        )
        if use_history:
            params["usehistory"] = "y"
        return f"{self.base_url.rstrip('/')}/esearch.fcgi?{_query_string(params)}"

    def build_esummary_url(
        self,
        pubmed_ids: list[str] | tuple[str, ...],
        *,
        include_credentials: bool = False,
    ) -> str:
        params = self._default_params(include_credentials=include_credentials)
        params.update({"db": "pubmed", "id": ",".join(pubmed_ids)})
        return f"{self.base_url.rstrip('/')}/esummary.fcgi?{_query_string(params)}"

    def build_efetch_url(
        self,
        pubmed_ids: list[str] | tuple[str, ...],
        *,
        retmode: str = "xml",
        rettype: str | None = None,
        include_credentials: bool = False,
    ) -> str:
        params: dict[str, Any] = {"db": "pubmed", "id": ",".join(pubmed_ids), "retmode": retmode}
        if include_credentials and self.email:
            params["email"] = self.email
        if include_credentials and self.api_key:
            params["api_key"] = self.api_key
        if self.tool:
            params["tool"] = self.tool
        if rettype:
            params["rettype"] = rettype
        return f"{self.base_url.rstrip('/')}/efetch.fcgi?{_query_string(params)}"

    def fetch_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object response")
        return payload

    def fetch_text(self, url: str) -> str:
        request = Request(url, headers={"Accept": "application/xml, text/xml, text/plain"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8")

    def extract_ids(self, payload: dict[str, Any]) -> list[str]:
        search_result = payload.get("esearchresult", {})
        ids = search_result.get("idlist", [])
        if not isinstance(ids, list):
            return []
        return [str(pmid) for pmid in ids]

    def normalize_summaries(self, payload: dict[str, Any]) -> list[PubMedArticle]:
        result = payload.get("result", {})
        uids = result.get("uids", [])
        if not isinstance(result, dict) or not isinstance(uids, list):
            return []

        articles: list[PubMedArticle] = []
        for uid in uids:
            item = result.get(str(uid), {})
            if not isinstance(item, dict):
                continue

            authors = [
                str(author.get("name", "")).strip()
                for author in item.get("authors", [])
                if isinstance(author, dict) and author.get("name")
            ]
            doi = ""
            for article_id in item.get("articleids", []):
                if isinstance(article_id, dict) and article_id.get("idtype") == "doi":
                    doi = str(article_id.get("value", "")).strip()
                    break

            pmid = str(item.get("uid", uid))
            articles.append(
                PubMedArticle(
                    pmid=pmid,
                    title=str(item.get("title", "")).strip(),
                    pubdate=str(item.get("pubdate", "")).strip(),
                    journal=str(item.get("fulljournalname", "")).strip(),
                    doi=doi,
                    authors=authors,
                    article_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    metadata={
                        "source": "pubmed",
                        "article_ids": item.get("articleids", []),
                        "pubtype": item.get("pubtype", []),
                    },
                )
            )
        return articles

    def parse_efetch_xml(self, xml_text: str) -> list[PubMedArticle]:
        root = ET.fromstring(xml_text)
        articles: list[PubMedArticle] = []

        for article_node in root.findall(".//PubmedArticle"):
            pmid = self._node_text(article_node.find("./MedlineCitation/PMID"))
            article = article_node.find("./MedlineCitation/Article")
            if article is None or not pmid:
                continue

            title = self._collapsed_text(article.find("./ArticleTitle"))
            journal = self._collapsed_text(article.find("./Journal/Title"))
            pubdate = self._extract_pubdate(article)
            abstract = self._extract_abstract(article)
            doi = ""

            article_id_list = article_node.find("./PubmedData/ArticleIdList")
            if article_id_list is not None:
                for article_id in article_id_list.findall("./ArticleId"):
                    if article_id.attrib.get("IdType") == "doi":
                        doi = self._collapsed_text(article_id)
                        break

            authors: list[str] = []
            for author in article.findall("./AuthorList/Author"):
                last_name = self._node_text(author.find("./LastName"))
                initials = self._node_text(author.find("./Initials"))
                collective = self._node_text(author.find("./CollectiveName"))
                author_name = " ".join(part for part in (last_name, initials) if part).strip()
                if author_name:
                    authors.append(author_name)
                elif collective:
                    authors.append(collective)

            articles.append(
                PubMedArticle(
                    pmid=pmid,
                    title=title,
                    pubdate=pubdate,
                    journal=journal,
                    doi=doi,
                    authors=authors,
                    article_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    abstract=abstract,
                    metadata={"source": "pubmed"},
                )
            )

        return articles

    def fetch_abstracts(self, pubmed_ids: list[str] | tuple[str, ...]) -> list[PubMedArticle]:
        xml_text = self.fetch_text(
            self.build_efetch_url(pubmed_ids, include_credentials=True, retmode="xml")
        )
        return self.parse_efetch_xml(xml_text)

    def fetch_articles(self, pubmed_ids: list[str] | tuple[str, ...]) -> list[PubMedArticle]:
        if not pubmed_ids:
            return []

        summary_payload = self.fetch_json(self.build_esummary_url(pubmed_ids, include_credentials=True))
        summaries = {article.pmid: article for article in self.normalize_summaries(summary_payload)}
        abstracts = {article.pmid: article for article in self.fetch_abstracts(pubmed_ids)}

        ordered_articles: list[PubMedArticle] = []
        for pmid in pubmed_ids:
            summary = summaries.get(str(pmid))
            abstract = abstracts.get(str(pmid))
            if summary is None and abstract is None:
                continue
            ordered_articles.append(
                PubMedArticle(
                    pmid=str(pmid),
                    title=(summary.title if summary else "") or (abstract.title if abstract else ""),
                    pubdate=(summary.pubdate if summary else "") or (abstract.pubdate if abstract else ""),
                    journal=(summary.journal if summary else "") or (abstract.journal if abstract else ""),
                    doi=(summary.doi if summary else "") or (abstract.doi if abstract else ""),
                    authors=(summary.authors if summary else []) or (abstract.authors if abstract else []),
                    article_url=(summary.article_url if summary else "") or (abstract.article_url if abstract else ""),
                    abstract=abstract.abstract if abstract else "",
                    metadata=(summary.metadata if summary else {"source": "pubmed"}),
                )
            )
        return ordered_articles

    def search_and_fetch_abstracts(
        self,
        term: str,
        *,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> list[PubMedArticle]:
        summaries = self.search_and_summarize(term, retmax=retmax, sort=sort)
        ids = [article.pmid for article in summaries]
        if not ids:
            return []
        return self.fetch_articles(ids)

    def search_and_summarize(
        self,
        term: str,
        *,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> list[PubMedArticle]:
        search_payload = self.fetch_json(
            self.build_esearch_url(
                term,
                retmax=retmax,
                sort=sort,
                include_credentials=True,
            )
        )
        ids = self.extract_ids(search_payload)
        if not ids:
            return []
        summary_payload = self.fetch_json(self.build_esummary_url(ids, include_credentials=True))
        return self.normalize_summaries(summary_payload)

    @staticmethod
    def _node_text(node: ET.Element | None) -> str:
        if node is None or node.text is None:
            return ""
        return node.text.strip()

    @staticmethod
    def _collapsed_text(node: ET.Element | None) -> str:
        if node is None:
            return ""
        return " ".join("".join(node.itertext()).split())

    def _extract_pubdate(self, article_node: ET.Element) -> str:
        pubdate = article_node.find("./Journal/JournalIssue/PubDate")
        if pubdate is None:
            return ""

        medline_date = self._node_text(pubdate.find("./MedlineDate"))
        if medline_date:
            return medline_date

        parts = [
            self._node_text(pubdate.find("./Year")),
            self._node_text(pubdate.find("./Month")),
            self._node_text(pubdate.find("./Day")),
        ]
        return " ".join(part for part in parts if part)

    def _extract_abstract(self, article_node: ET.Element) -> str:
        abstract_nodes = article_node.findall("./Abstract/AbstractText")
        parts: list[str] = []
        for abstract_node in abstract_nodes:
            text = self._collapsed_text(abstract_node)
            if not text:
                continue
            label = abstract_node.attrib.get("Label") or abstract_node.attrib.get("NlmCategory")
            if label and not text.lower().startswith(label.lower()):
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)
        return "\n".join(parts)
