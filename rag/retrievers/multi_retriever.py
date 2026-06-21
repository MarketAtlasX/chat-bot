from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from rag.retrievers.base import BaseRetriever, RetrievalResult, RetrieverType
from rag.retrievers.graph_retriever import GraphRetriever
from rag.retrievers.historical_retriever import HistoricalRetriever
from rag.retrievers.market_retriever import MarketRetriever
from rag.retrievers.news_retriever import NewsRetriever

logger = logging.getLogger(__name__)


@dataclass
class MultiRetrievalResult:
    query: str
    results: Dict[RetrieverType, List[RetrievalResult]] = field(default_factory=dict)
    all_results: List[RetrievalResult] = field(default_factory=list)


class MultiRetriever:
    def __init__(self, retrievers: Optional[Dict[RetrieverType, BaseRetriever]] = None):
        self.retrievers = retrievers or {
            RetrieverType.NEWS: NewsRetriever(),
            RetrieverType.HISTORICAL: HistoricalRetriever(),
            RetrieverType.GRAPH: GraphRetriever(),
            RetrieverType.MARKET: MarketRetriever(),
        }

    async def retrieve_all(
        self,
        query: str,
        limit_per_source: int = 5,
        include_types: Optional[List[RetrieverType]] = None,
    ) -> MultiRetrievalResult:
        types_to_run = include_types or list(self.retrievers.keys())
        tasks = {}
        for rt in types_to_run:
            if rt in self.retrievers:
                tasks[rt] = self.retrievers[rt].retrieve(query, limit=limit_per_source)
        completed = {}
        for rt, task in tasks.items():
            try:
                completed[rt] = await task
            except Exception as e:
                logger.error(f"Retriever {rt} failed: {e}")
                completed[rt] = []
        all_results = []
        for rt, results in completed.items():
            all_results.extend(results)
        all_results.sort(key=lambda r: r.score, reverse=True)
        return MultiRetrievalResult(
            query=query,
            results=completed,
            all_results=all_results,
        )

    async def retrieve_filtered(
        self,
        query: str,
        include_types: List[RetrieverType],
        limit_per_source: int = 5,
        score_threshold: Optional[float] = None,
    ) -> MultiRetrievalResult:
        return await self.retrieve_all(
            query=query,
            limit_per_source=limit_per_source,
            include_types=include_types,
        )
