from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from rag.historical_memory.event_embeddings import EventEmbedder, HISTORICAL_EVENTS


@dataclass
class SimilarityScore:
    event_name: str
    text_similarity: float = 0.0
    entity_similarity: float = 0.0
    sector_similarity: float = 0.0
    region_similarity: float = 0.0
    combined_score: float = 0.0
    event_data: dict = field(default_factory=dict)


class EventSimilarity:
    def __init__(self):
        self.embedder = EventEmbedder()

    def compute_similarity(
        self,
        query: str,
        event: dict,
        text_weight: float = 0.4,
        entity_weight: float = 0.3,
        sector_weight: float = 0.2,
        region_weight: float = 0.1,
    ) -> SimilarityScore:
        query_lower = query.lower()
        event_text = f"{event['name']} {event['description']}".lower()
        query_tokens = set(query_lower.split())
        event_tokens = set(event_text.split())
        if not query_tokens or not event_tokens:
            text_sim = 0.0
        else:
            text_sim = len(query_tokens & event_tokens) / len(query_tokens | event_tokens)
        query_entities = {e.lower() for e in query_lower.split()}
        event_entities = {e.lower() for e in event.get("entities", [])}
        entity_sim = 0.0
        if query_entities and event_entities:
            entity_sim = len(query_entities & event_entities) / len(query_entities | event_entities)
        else:
            for entity in event.get("entities", []):
                if entity.lower() in query_lower:
                    entity_sim += 0.2
            entity_sim = min(entity_sim, 1.0)
        sector_sim = 0.0
        event_sectors = [s.lower() for s in event.get("sectors", [])]
        if event_sectors:
            matches = sum(1 for s in event_sectors if s in query_lower)
            sector_sim = matches / len(event_sectors)
        region_sim = 0.0
        event_regions = [r.lower() for r in event.get("regions", [])]
        if event_regions:
            matches = sum(1 for r in event_regions if r in query_lower)
            region_sim = matches / len(event_regions)
        combined = (
            text_sim * text_weight
            + entity_sim * entity_weight
            + sector_sim * sector_weight
            + region_sim * region_weight
        )
        return SimilarityScore(
            event_name=event["name"],
            text_similarity=text_sim,
            entity_similarity=entity_sim,
            sector_similarity=sector_sim,
            region_similarity=region_sim,
            combined_score=combined,
            event_data=event,
        )

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> List[SimilarityScore]:
        results = []
        query_vec = self.embedder.embed_text(query)
        for event in HISTORICAL_EVENTS:
            event_vec = self.embedder.get_embedding(event["name"])
            text_sim = 0.0
            if event_vec is not None and query_vec is not None:
                cos_sim = np.dot(query_vec, event_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(event_vec) + 1e-8
                )
                text_sim = float(max(0, cos_sim))
            keyword_result = self.compute_similarity(query, event)
            combined_text = (text_sim + keyword_result.text_similarity) / 2
            combined = (
                combined_text * 0.4
                + keyword_result.entity_similarity * 0.3
                + keyword_result.sector_similarity * 0.2
                + keyword_result.region_similarity * 0.1
            )
            if combined >= threshold:
                results.append(
                    SimilarityScore(
                        event_name=event["name"],
                        text_similarity=combined_text,
                        entity_similarity=keyword_result.entity_similarity,
                        sector_similarity=keyword_result.sector_similarity,
                        region_similarity=keyword_result.region_similarity,
                        combined_score=combined,
                        event_data=event,
                    )
                )
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
