from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from rag.embeddings import get_embedding_model
from rag.ingestion import Document, NewsIngestor
from rag.retrievers.news_retriever import NewsRetriever
from rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class NewsRAGResult:
    query: str
    results: list = field(default_factory=list)
    context: str = ""
    sources: List[str] = field(default_factory=list)


class NewsRAGPipeline:
    def __init__(self, collection: str = "marketatlas_news"):
        self.collection = collection
        self.ingestor = NewsIngestor(collection=collection)
        self.retriever = NewsRetriever(collection=collection)
        self.embedder = get_embedding_model()

    def ingest(self, document: Document) -> bool:
        result = self.ingestor.ingest(document)
        return result.success

    def ingest_batch(self, documents: List[Document]) -> List[bool]:
        return [r.success for r in self.ingestor.ingest_batch(documents)]

    async def query(
        self,
        question: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> NewsRAGResult:
        results = await self.retriever.retrieve(
            query=question,
            limit=limit,
            score_threshold=score_threshold,
        )
        sources = list(set(r.metadata.get("source", "unknown") for r in results))
        context_parts = []
        for i, r in enumerate(results, 1):
            title = r.metadata.get("title", r.metadata.get("name", "Untitled"))
            context_parts.append(f"[{i}] {title} (relevance: {r.score:.2f})")
            context_parts.append(f"    {r.content[:500]}")
        context = "\n".join(context_parts) if context_parts else "No relevant news found."
        return NewsRAGResult(
            query=question,
            results=[{"content": r.content, "score": r.score, "source": r.source, "metadata": r.metadata} for r in results],
            context=context,
            sources=sources,
        )
