from __future__ import annotations

import logging
from typing import List, Optional

from rag.embeddings import get_embedding_model
from rag.retrievers.base import BaseRetriever, RetrievalResult, RetrieverType
from rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


class NewsRetriever(BaseRetriever):
    def __init__(self, collection: str = "marketatlas_news"):
        super().__init__(name="news_retriever", retriever_type=RetrieverType.NEWS)
        self.collection = collection
        self.embedder = get_embedding_model()
        self.store = get_vector_store(collection)

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict] = None,
    ) -> List[RetrievalResult]:
        query_vec = self.embedder.embed_query(query)
        results = self.store.search(
            query_vector=query_vec,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )
        return [
            RetrievalResult(
                content=r.payload.get("text", ""),
                score=r.score,
                source=r.payload.get("source", "news"),
                retriever_type=RetrieverType.NEWS,
                metadata={
                    "title": r.payload.get("title", ""),
                    "url": r.payload.get("url", ""),
                    "published_at": r.payload.get("published_at", ""),
                    "topics": r.payload.get("topics", ""),
                    "sentiment": r.payload.get("sentiment", ""),
                },
                id=r.id,
            )
            for r in results
        ]
