from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag.embeddings import get_embedding_model
from rag.retrievers.base import BaseRetriever, RetrievalResult, RetrieverType
from rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


MARKET_REACTIONS = [
    {
        "event": "Iran sanctions 2018",
        "asset": "Brent Crude",
        "reaction": "prices surged 25% over 3 months following US withdrawal from JCPOA",
        "impact": "high",
        "sector": "energy",
        "recovery": "partial after 6 months",
    },
    {
        "event": "Russia-Ukraine war 2022",
        "asset": "European natural gas",
        "reaction": "prices increased 400% within 6 months of invasion",
        "impact": "severe",
        "sector": "energy",
        "recovery": "gradual over 18 months",
    },
    {
        "event": "Taiwan strait crisis 2022",
        "asset": "Semiconductor stocks (SOX)",
        "reaction": "declined 15% during Pelosi visit, recovered in 2 weeks",
        "impact": "moderate",
        "sector": "technology",
        "recovery": "quick within 1 month",
    },
    {
        "event": "Hormuz strait disruption 2019",
        "asset": "Brent Crude",
        "reaction": "prices spiked 15% in 1 week after tanker attacks",
        "impact": "high",
        "sector": "energy",
        "recovery": "partial after OPEC response",
    },
    {
        "event": "US-China trade war 2019",
        "asset": "S&P 500",
        "reaction": "declined 10% over 3 months during tariff escalation",
        "impact": "moderate",
        "sector": "broad market",
        "recovery": "V-shaped recovery over 4 months",
    },
    {
        "event": "Gulf war 1990",
        "asset": "Brent Crude",
        "reaction": "prices doubled from $15 to $30 in 3 months",
        "impact": "severe",
        "sector": "energy",
        "recovery": "6 months post ceasefire",
    },
    {
        "event": "COVID-19 pandemic 2020",
        "asset": "S&P 500",
        "reaction": "declined 34% in 1 month, then recovered over 6 months",
        "impact": "severe",
        "sector": "broad market",
        "recovery": "V-shaped, full recovery in 6 months",
    },
    {
        "event": "Libya civil war 2011",
        "asset": "Brent Crude",
        "reaction": "prices rose 25% during peak conflict",
        "impact": "moderate",
        "sector": "energy",
        "recovery": "gradual over 4 months",
    },
    {
        "event": "Korean peninsula tensions 2017",
        "asset": "KOSPI",
        "reaction": "declined 5% during missile tests, recovered quickly",
        "impact": "low",
        "sector": "equities",
        "recovery": "quick within weeks",
    },
    {
        "event": "Suez canal blockage 2021",
        "asset": "Shipping rates",
        "reaction": "freight rates increased 300% in 1 week",
        "impact": "moderate",
        "sector": "shipping",
        "recovery": "2 weeks after clearing",
    },
]


def seed_market_reactions(collection: str = "marketatlas_market"):
    from rag.embeddings import get_embedding_model
    from rag.vectorstore import get_vector_store
    embedder = get_embedding_model()
    store = get_vector_store(collection)
    texts = []
    metadatas = []
    for reaction in MARKET_REACTIONS:
        text = f"Event: {reaction['event']}\nAsset: {reaction['asset']}\nReaction: {reaction['reaction']}\nImpact: {reaction['impact']}\nSector: {reaction['sector']}\nRecovery: {reaction['recovery']}"
        texts.append(text)
        metadatas.append(reaction)
    vectors = embedder.embed(texts)
    store.add(texts, vectors, metadatas)
    return len(texts)


class MarketRetriever(BaseRetriever):
    def __init__(self, collection: str = "marketatlas_market"):
        super().__init__(name="market_retriever", retriever_type=RetrieverType.MARKET)
        self.collection = collection
        self.embedder = get_embedding_model()
        self.store = get_vector_store(collection)

    async def retrieve(
        self,
        query: str,
        limit: int = 5,
        **kwargs,
    ) -> List[RetrievalResult]:
        results = []
        query_lower = query.lower()
        scored = []
        for reaction in MARKET_REACTIONS:
            score = 0.0
            event_text = f"{reaction['event']} {reaction['asset']} {reaction['reaction']} {reaction['sector']}"
            query_terms = query_lower.split()
            event_terms = event_text.lower().split()
            matches = sum(1 for qt in query_terms if qt in event_text.lower())
            score = matches / max(len(query_terms), 1) * 0.8
            entity_matches = 0
            if reaction["sector"] in query_lower:
                entity_matches += 0.3
            if reaction["asset"].lower() in query_lower:
                entity_matches += 0.4
            score = min(score + entity_matches, 1.0)
            if score > 0.1:
                scored.append((score, reaction))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, reaction in scored[:limit]:
            results.append(
                RetrievalResult(
                    content=f"Event: {reaction['event']}\nAsset: {reaction['asset']}\nReaction: {reaction['reaction']}\nImpact: {reaction['impact']}\nSector: {reaction['sector']}\nRecovery: {reaction['recovery']}",
                    score=score,
                    source="market_reactions",
                    retriever_type=RetrieverType.MARKET,
                    metadata=reaction,
                )
            )
        if not results:
            query_vec = self.embedder.embed_query(query)
            vec_results = self.store.search(query_vector=query_vec, limit=limit)
            for r in vec_results:
                results.append(
                    RetrievalResult(
                        content=r.payload.get("text", ""),
                        score=r.score * 0.9,
                        source=r.payload.get("source", "market"),
                        retriever_type=RetrieverType.MARKET,
                        metadata=r.payload,
                        id=r.id,
                    )
                )
        return results
