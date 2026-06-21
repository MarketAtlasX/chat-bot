from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from rag.historical_memory.analog_retriever import AnalogRetriever, AnalogResult
from rag.historical_memory.event_similarity import EventSimilarity, SimilarityScore
from rag.retrievers.historical_retriever import HistoricalRetriever

logger = logging.getLogger(__name__)


@dataclass
class HistoricalSimilarityResult:
    query: str
    analogs: List[dict] = field(default_factory=list)
    context: str = ""
    summary: str = ""


class HistoricalSimilarityPipeline:
    def __init__(self):
        self.analog_retriever = AnalogRetriever()
        self.event_similarity = EventSimilarity()
        self.event_retriever = HistoricalRetriever()

    def find_analogs(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> HistoricalSimilarityResult:
        analog_result = self.analog_retriever.find_analogs(
            query=query,
            top_k=top_k,
            threshold=threshold,
        )
        context_parts = []
        for a in analog_result.analogs:
            event = a.event_data
            context_parts.append(
                f"Event: {a.event_name}\n"
                f"Similarity Score: {a.combined_score:.2f}\n"
                f"Description: {event.get('description', '')}\n"
                f"Impact: {event.get('impact', 'N/A')}\n"
                f"Market Effect: {event.get('market_effect', 'N/A')}\n"
                f"Entities: {', '.join(event.get('entities', []))}\n"
                f"Sectors: {', '.join(event.get('sectors', []))}\n"
                f"Regions: {', '.join(event.get('regions', []))}\n"
            )
        context = "\n---\n".join(context_parts) if context_parts else "No historical analogs found."
        return HistoricalSimilarityResult(
            query=query,
            analogs=[
                {
                    "event_name": a.event_name,
                    "score": a.combined_score,
                    "text_similarity": a.text_similarity,
                    "entity_similarity": a.entity_similarity,
                    "sector_similarity": a.sector_similarity,
                    "region_similarity": a.region_similarity,
                    "description": a.event_data.get("description", ""),
                    "market_effect": a.event_data.get("market_effect", ""),
                    "impact": a.event_data.get("impact", ""),
                    "entities": a.event_data.get("entities", []),
                    "sectors": a.event_data.get("sectors", []),
                    "regions": a.event_data.get("regions", []),
                }
                for a in analog_result.analogs
            ],
            context=context,
            summary=analog_result.summary if analog_result.analogs else "No significant historical analogs found.",
        )

    async def query(self, question: str, top_k: int = 5) -> HistoricalSimilarityResult:
        return self.find_analogs(question, top_k=top_k)
