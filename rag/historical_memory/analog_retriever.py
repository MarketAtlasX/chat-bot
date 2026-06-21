from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from rag.historical_memory.event_similarity import EventSimilarity, SimilarityScore
from rag.retrievers.base import RetrievalResult, RetrieverType

logger = logging.getLogger(__name__)


@dataclass
class AnalogResult:
    query: str
    analogs: List[SimilarityScore] = field(default_factory=list)
    summary: str = ""

    def to_retrieval_results(self) -> List[RetrievalResult]:
        return [
            RetrievalResult(
                content=f"Historical Analog: {a.event_name}\nSimilarity: {a.combined_score:.2f}\nText Match: {a.text_similarity:.2f}, Entity Match: {a.entity_similarity:.2f}, Sector Match: {a.sector_similarity:.2f}, Region Match: {a.region_similarity:.2f}\nDescription: {a.event_data.get('description', '')}\nMarket Effect: {a.event_data.get('market_effect', 'N/A')}",
                score=a.combined_score,
                source="historical_analog",
                retriever_type=RetrieverType.HISTORICAL,
                metadata={
                    "event_name": a.event_name,
                    "text_similarity": a.text_similarity,
                    "entity_similarity": a.entity_similarity,
                    "sector_similarity": a.sector_similarity,
                    "region_similarity": a.region_similarity,
                    "combined_score": a.combined_score,
                    **a.event_data,
                },
            )
            for a in self.analogs
        ]


class AnalogRetriever:
    def __init__(self):
        self.similarity = EventSimilarity()

    def find_analogs(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> AnalogResult:
        analogs = self.similarity.find_similar(query, top_k=top_k, threshold=threshold)
        if analogs:
            summary = f"Found {len(analogs)} historical analogs for: '{query}'. Top match: {analogs[0].event_name} (score: {analogs[0].combined_score:.2f})"
        else:
            summary = f"No significant historical analogs found for: '{query}'."
        return AnalogResult(query=query, analogs=analogs, summary=summary)
