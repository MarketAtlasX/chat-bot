from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag.retrievers.market_retriever import MarketRetriever, MARKET_REACTIONS
from rag.historical_memory.event_similarity import EventSimilarity

logger = logging.getLogger(__name__)


@dataclass
class MarketRAGResult:
    query: str
    reactions: List[dict] = field(default_factory=list)
    context: str = ""
    summary: str = ""
    assets_impacted: List[str] = field(default_factory=list)
    sectors_impacted: List[str] = field(default_factory=list)


class MarketRAGPipeline:
    def __init__(self):
        self.market_retriever = MarketRetriever()
        self.event_similarity = EventSimilarity()

    async def query(
        self,
        question: str,
        limit: int = 5,
    ) -> MarketRAGResult:
        results = await self.market_retriever.retrieve(query=question, limit=limit)

        reactions = [
            {
                "event": r.metadata.get("event", ""),
                "asset": r.metadata.get("asset", ""),
                "reaction": r.metadata.get("reaction", ""),
                "impact": r.metadata.get("impact", ""),
                "sector": r.metadata.get("sector", ""),
                "recovery": r.metadata.get("recovery", ""),
                "score": r.score,
                "content": r.content,
            }
            for r in results
        ]

        context_parts = []
        for r in results:
            title = r.metadata.get("event", r.metadata.get("title", "Unknown Event"))
            context_parts.append(f"Event: {title} (relevance: {r.score:.2f})")
            context_parts.append(f"  {r.content[:500]}")
        context = "\n".join(context_parts) if context_parts else "No market data found."

        assets = list(set(r.metadata.get("asset", "") for r in results if r.metadata.get("asset")))
        sectors = list(set(r.metadata.get("sector", "") for r in results if r.metadata.get("sector")))

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        if reactions:
            summary = f"Found {len(reactions)} market reactions (avg relevance: {avg_score:.2f}). " \
                      f"Top match: {reactions[0]['event']} affecting {reactions[0]['asset']}."
        else:
            summary = "No relevant market reactions found."

        return MarketRAGResult(
            query=question,
            reactions=reactions,
            context=context,
            summary=summary,
            assets_impacted=assets,
            sectors_impacted=sectors,
        )
