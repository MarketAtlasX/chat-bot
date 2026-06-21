from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag.geo_rag.context_builder import GeoContext, GeoContextBuilder
from rag.geo_rag.intent_classifier import GeoIntent, GeoIntentClassifier, IntentResult
from rag.historical_memory.analog_retriever import AnalogRetriever
from rag.rerankers import get_reranker
from rag.retrievers.base import RetrievalResult, RetrieverType
from rag.retrievers.graph_retriever import GraphRetriever
from rag.retrievers.historical_retriever import HistoricalRetriever
from rag.retrievers.market_retriever import MarketRetriever
from rag.retrievers.multi_retriever import MultiRetriever, MultiRetrievalResult
from rag.retrievers.news_retriever import NewsRetriever

logger = logging.getLogger(__name__)


@dataclass
class GeoRAGResult:
    query: str
    intent: IntentResult
    context: GeoContext
    multi_results: Optional[MultiRetrievalResult] = None
    reranked_results: List[RetrievalResult] = field(default_factory=list)
    all_results: List[RetrievalResult] = field(default_factory=list)
    prompt: str = ""
    sources_used: List[str] = field(default_factory=list)


class GeoRAGPipeline:
    def __init__(
        self,
        news_retriever: Optional[NewsRetriever] = None,
        historical_retriever: Optional[HistoricalRetriever] = None,
        graph_retriever: Optional[GraphRetriever] = None,
        market_retriever: Optional[MarketRetriever] = None,
        analog_retriever: Optional[AnalogRetriever] = None,
        multi_retriever: Optional[MultiRetriever] = None,
        intent_classifier: Optional[GeoIntentClassifier] = None,
        context_builder: Optional[GeoContextBuilder] = None,
    ):
        self.news_retriever = news_retriever or NewsRetriever()
        self.historical_retriever = historical_retriever or HistoricalRetriever()
        self.graph_retriever = graph_retriever or GraphRetriever()
        self.market_retriever = market_retriever or MarketRetriever()
        self.analog_retriever = analog_retriever or AnalogRetriever()
        self.multi_retriever = multi_retriever or MultiRetriever()
        self.intent_classifier = intent_classifier or GeoIntentClassifier()
        self.context_builder = context_builder or GeoContextBuilder()
        self.reranker = get_reranker()

    async def run(
        self,
        query: str,
        intent_override: Optional[GeoIntent] = None,
        limit_per_source: int = 5,
        use_reranker: bool = True,
    ) -> GeoRAGResult:
        intent = self.intent_classifier.classify(query)
        if intent_override:
            intent.intent = intent_override

        news_results: List[RetrievalResult] = []
        historical_results: List[RetrievalResult] = []
        graph_results: List[RetrievalResult] = []
        market_results: List[RetrievalResult] = []

        retrievers_to_run = self._get_retrievers_for_intent(intent.intent)

        async def run_news():
            nonlocal news_results
            news_results = await self.news_retriever.retrieve(query, limit=limit_per_source)

        async def run_historical():
            nonlocal historical_results
            historical_results = await self.historical_retriever.retrieve(query, limit=limit_per_source)
            analogs = self.analog_retriever.find_analogs(query, top_k=limit_per_source)
            analog_results = analogs.to_retrieval_results()
            existing_names = {r.metadata.get("event_name", "") for r in historical_results}
            for ar in analog_results:
                name = ar.metadata.get("event_name", "")
                if name not in existing_names:
                    historical_results.append(ar)
                    existing_names.add(name)
            historical_results.sort(key=lambda r: r.score, reverse=True)
            historical_results = historical_results[:limit_per_source]

        async def run_graph():
            nonlocal graph_results
            graph_results = await self.graph_retriever.retrieve(query, limit=limit_per_source)

        async def run_market():
            nonlocal market_results
            market_results = await self.market_retriever.retrieve(query, limit=limit_per_source)

        tasks = []
        if "news" in retrievers_to_run:
            tasks.append(run_news())
        if "historical" in retrievers_to_run or "analog" in retrievers_to_run:
            tasks.append(run_historical())
        if "graph" in retrievers_to_run:
            tasks.append(run_graph())
        if "market" in retrievers_to_run:
            tasks.append(run_market())

        if tasks:
            await asyncio.gather(*tasks)

        all_results = news_results + historical_results + graph_results + market_results
        all_results.sort(key=lambda r: r.score, reverse=True)

        reranked = all_results
        if use_reranker and len(all_results) > 1:
            try:
                documents = [r.content for r in all_results]
                metadatas = [r.metadata for r in all_results]
                ids = [r.id for r in all_results]
                rerank_results = self.reranker.rerank(
                    query=query,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    top_k=limit_per_source * 2,
                )
                reranked = [
                    RetrievalResult(
                        content=rr.content,
                        score=rr.score,
                        source="reranked",
                        retriever_type=RetrieverType.NEWS,
                        metadata=rr.metadata,
                        id=rr.id,
                    )
                    for rr in rerank_results
                ]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        context = self.context_builder.build(
            query=query,
            news_results=news_results,
            historical_results=historical_results,
            graph_results=graph_results,
            market_results=market_results,
        )

        prompt = self._build_prompt(query, context, intent)

        sources_used = []
        if news_results:
            sources_used.append("news")
        if historical_results:
            sources_used.append("historical")
        if graph_results:
            sources_used.append("graph")
        if market_results:
            sources_used.append("market")

        multi_results = MultiRetrievalResult(
            query=query,
            results={
                RetrieverType.NEWS: news_results,
                RetrieverType.HISTORICAL: historical_results,
                RetrieverType.GRAPH: graph_results,
                RetrieverType.MARKET: market_results,
            },
            all_results=all_results,
        )

        return GeoRAGResult(
            query=query,
            intent=intent,
            context=context,
            multi_results=multi_results,
            reranked_results=reranked,
            all_results=all_results,
            prompt=prompt,
            sources_used=sources_used,
        )

    def _get_retrievers_for_intent(self, intent: GeoIntent) -> List[str]:
        intent_map = {
            GeoIntent.NEWS_ANALYSIS: ["news"],
            GeoIntent.HISTORICAL_ANALOG: ["historical", "analog"],
            GeoIntent.MARKET_IMPACT: ["market", "news"],
            GeoIntent.GRAPH_RELATIONSHIP: ["graph"],
            GeoIntent.GEOPOLITICAL_RISK: ["news", "historical", "graph"],
            GeoIntent.GENERAL_QUERY: ["news", "historical"],
            GeoIntent.MULTI_SOURCE: ["news", "historical", "graph", "market"],
        }
        return intent_map.get(intent, ["news"])

    def _build_prompt(self, query: str, context: GeoContext, intent: IntentResult) -> str:
        prompt_parts = [
            "You are MarketAtlas GeoRAG, a geopolitical analysis AI. Answer the question based on the retrieved context.",
            f"\nIntent: {intent.intent.value} (confidence: {intent.confidence:.2f})",
        ]
        if intent.extracted_entities:
            prompt_parts.append(f"Entities detected: {', '.join(intent.extracted_entities)}")
        if intent.extracted_sectors:
            prompt_parts.append(f"Sectors detected: {', '.join(intent.extracted_sectors)}")
        if intent.extracted_regions:
            prompt_parts.append(f"Regions detected: {', '.join(intent.extracted_regions)}")
        if context.source_summary:
            prompt_parts.append(f"\nContext Sources: {context.source_summary}")
        if context.combined_text:
            prompt_parts.append(f"\nRetrieved Context:\n{context.combined_text}")
        prompt_parts.append(f"\nQuestion: {query}")
        prompt_parts.append("\nProvide a comprehensive, well-reasoned answer based on the context above. Cite specific sources where relevant. If the context is insufficient, acknowledge this and provide your best analysis.")
        return "\n".join(prompt_parts)
