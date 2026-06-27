from __future__ import annotations
import abc
from typing import List

class EvidenceTool(abc.ABC):

    @abc.abstractmethod
    def search(self, query: str, k: int=3) -> List[str]:
        raise NotImplementedError

class InMemoryRetriever(EvidenceTool):

    def __init__(self, documents: List[str]) -> None:
        self._docs = documents

    def search(self, query: str, k: int=3) -> List[str]:
        terms = {w.lower() for w in query.split() if len(w) > 2}
        scored = []
        for doc in self._docs:
            text = doc.lower()
            score = sum((text.count(t) for t in terms))
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [doc for _, doc in scored[:k]]

class DuckDuckGoSearch(EvidenceTool):
    URL = 'https://html.duckduckgo.com/html/'

    def search(self, query: str, k: int=3) -> List[str]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []
        try:
            resp = requests.post(self.URL, data={'q': query}, headers={'User-Agent': 'Mozilla/5.0 (X-Agent)'}, timeout=10)
            resp.raise_for_status()
        except Exception:
            return []
        soup = BeautifulSoup(resp.text, 'html.parser')
        snippets = []
        for node in soup.select('.result__snippet')[:k]:
            text = node.get_text(strip=True)
            if text:
                snippets.append(text)
        return snippets
