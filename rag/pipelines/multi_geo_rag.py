from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.geo_rag.main_pipeline import GeoRAGPipeline, GeoRAGResult

logger = logging.getLogger(__name__)


@dataclass
class MultiGeoRAGResult:
    query: str
    intent: str = ""
    confidence: float = 0.0
    news_context: str = ""
    historical_context: str = ""
    graph_context: str = ""
    market_context: str = ""
    combined_context: str = ""
    reranked_context: str = ""
    prompt: str = ""
    sources_used: List[str] = field(default_factory=list)
    total_results: int = 0
    entities_detected: List[str] = field(default_factory=list)
    sectors_detected: List[str] = field(default_factory=list)
    regions_detected: List[str] = field(default_factory=list)


class MultiGeoRAGPipeline:
    def __init__(self, geo_pipeline: Optional[GeoRAGPipeline] = None):
        self.geo_pipeline = geo_pipeline or GeoRAGPipeline()

    async def query(
        self,
        question: str,
        limit_per_source: int = 5,
        use_reranker: bool = True,
    ) -> MultiGeoRAGResult:
        result = await self.geo_pipeline.run(
            query=question,
            limit_per_source=limit_per_source,
            use_reranker=use_reranker,
        )
        reranked_context = ""
        if result.reranked_results:
            lines = ["=== Reranked Results ==="]
            for i, r in enumerate(result.reranked_results, 1):
                content_preview = r.content[:300]
                lines.append(f"[{i}] Score: {r.score:.3f}")
                lines.append(f"    {content_preview}")
            reranked_context = "\n".join(lines)
        return MultiGeoRAGResult(
            query=question,
            intent=result.intent.intent.value if result.intent else "unknown",
            confidence=result.intent.confidence if result.intent else 0.0,
            news_context=result.context.news_context,
            historical_context=result.context.historical_context,
            graph_context=result.context.graph_context,
            market_context=result.context.market_context,
            combined_context=result.context.combined_text,
            reranked_context=reranked_context,
            prompt=result.prompt,
            sources_used=result.sources_used,
            total_results=len(result.all_results),
            entities_detected=result.intent.extracted_entities if result.intent else [],
            sectors_detected=result.intent.extracted_sectors if result.intent else [],
            regions_detected=result.intent.extracted_regions if result.intent else [],
        )
