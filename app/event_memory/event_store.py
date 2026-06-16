import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

import re
import numpy as np
from typing import Optional, Any
from collections import defaultdict

from ..config import settings
from ..rag.embeddings import embedding_model
from .event_schema import HistoricalEvent, EventSimilarityResult, SimilarityResponse
from .event_data import seed_events


class EventStore:
    def __init__(self):
        self.client = None
        self.collection = "marketatlas_event_memory"
        self._events: dict[str, HistoricalEvent] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._tokenized_texts: dict[str, set[str]] = {}
        self._initialized = False
        self._has_real_embeddings = False

    def _connect(self):
        if self.client is not None:
            return
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            self.client = QdrantClient(url=settings.qdrant_url, timeout=5)
            self.client.get_collections()
            self._ensure_collection(models)
        except Exception:
            self.client = None

    def _ensure_collection(self, models=None):
        if not self.client:
            return
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=embedding_model.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception:
            pass

    @property
    def available(self) -> bool:
        self._connect()
        return self.client is not None

    def _initialize(self):
        if self._initialized:
            return
        self._initialized = True
        events = seed_events()
        for ev in events:
            self._events[ev.id] = ev
        self._compute_embeddings()

    def _tokenize(self, text: str) -> set[str]:
        tokens = set(re.findall(r'[a-zA-Z0-9]+', text.lower()))
        stopwords = {
            "the", "a", "an", "is", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "shall", "should", "may", "might", "must",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "this", "that", "these", "those",
            "it", "its", "which", "who", "whom", "what", "when", "where",
            "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "only", "own", "same",
            "so", "than", "too", "very", "just", "about", "above", "after",
            "again", "against", "below", "between", "during", "into",
            "over", "through", "under", "up", "upon", "out", "off",
            "also", "as", "because", "before", "while", "if", "then", "else",
        }
        return tokens - stopwords

    def _compute_embeddings(self):
        texts_with_ids = [(ev.id, ev.name + ". " + ev.description) for ev in self._events.values()]
        if not texts_with_ids:
            return
        ids, texts = zip(*texts_with_ids)
        vectors = embedding_model.encode(list(texts))
        for eid, vec in zip(ids, vectors):
            self._embeddings[eid] = vec
            self._tokenized_texts[eid] = self._tokenize(
                (self._events[eid].name + " " + self._events[eid].description)
            )

        if vectors and len(vectors) > 0:
            v = np.array(vectors[0])
            norm = np.linalg.norm(v)
            self._has_real_embeddings = abs(norm - 1.0) < 0.1 if norm > 0 else False

    def add_event(self, event: HistoricalEvent):
        self._initialize()
        self._events[event.id] = event
        vec = embedding_model.encode([event.name + ". " + event.description])[0]
        self._embeddings[event.id] = vec
        self._tokenized_texts[event.id] = self._tokenize(
            event.name + " " + event.description
        )
        self.upsert_events([event])

    def get_all_events(self) -> list[HistoricalEvent]:
        self._initialize()
        return list(self._events.values())

    def get_event(self, event_id: str) -> Optional[HistoricalEvent]:
        self._initialize()
        return self._events.get(event_id)

    def _jaccard_similarity(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        intersection = a & b
        union = a | b
        return len(intersection) / len(union)

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        a = np.array(vec_a, dtype=np.float64)
        b = np.array(vec_b, dtype=np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _keyword_text_similarity(self, query_tokens: set[str], event_id: str) -> float:
        event_tokens = self._tokenized_texts.get(event_id, set())
        if not query_tokens or not event_tokens:
            return 0.0
        overlap = query_tokens & event_tokens
        if not overlap:
            return 0.0
        recall = len(overlap) / len(query_tokens)
        precision = len(overlap) / len(event_tokens)
        if recall + precision == 0:
            return 0.0
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _compute_text_similarity(self, query_vec: list[float], query_tokens: set[str], event_id: str) -> float:
        if self._has_real_embeddings:
            event_vec = self._embeddings.get(event_id)
            if event_vec is not None:
                cos_sim = self._cosine_similarity(query_vec, event_vec)
                return max(0.0, cos_sim)
        return self._keyword_text_similarity(query_tokens, event_id)

    def _compute_entity_similarity(self, query_entities: set, event: HistoricalEvent) -> float:
        event_entities = set(e.lower() for e in event.entities)
        query_ents = set(e.lower() for e in query_entities)
        base = self._jaccard_similarity(query_ents, event_entities)

        if base > 0:
            return base

        if not query_ents or not event_entities:
            return 0.0

        word_overlap = 0
        for qe in query_ents:
            qe_words = set(qe.split())
            for ee in event_entities:
                ee_words = set(ee.split())
                if qe_words & ee_words:
                    word_overlap += 1
                    break

        if word_overlap > 0:
            return word_overlap / len(query_ents) * 0.5

        return 0.0

    def _compute_sector_similarity(self, query_sectors: set, event: HistoricalEvent) -> float:
        event_sectors = set(s.lower() for s in event.sectors)
        query_sects = set(s.lower() for s in query_sectors)
        return self._jaccard_similarity(query_sects, event_sectors)

    def _compute_market_similarity(self, query_outcome_sectors: set, event: HistoricalEvent) -> float:
        event_outcome_sectors = set(o.sector.lower() for o in event.outcomes)
        return self._jaccard_similarity(query_outcome_sectors, event_outcome_sectors)

    def find_similar(
        self,
        query_text: str,
        query_entities: Optional[list[str]] = None,
        query_sectors: Optional[list[str]] = None,
        query_outcome_sectors: Optional[list[str]] = None,
        top_k: int = 5,
    ) -> list[EventSimilarityResult]:
        self._initialize()

        query_vec = embedding_model.encode_query(query_text)
        query_tokens = self._tokenize(query_text)
        query_entities_set = set(query_entities or [])
        query_sectors_set = set(query_sectors or [])
        query_outcome_set = set(query_outcome_sectors or [])

        results = []
        for event in self._events.values():
            text_sim = self._compute_text_similarity(query_vec, query_tokens, event.id)
            entity_sim = self._compute_entity_similarity(query_entities_set, event)
            sector_sim = self._compute_sector_similarity(query_sectors_set, event)
            market_sim = self._compute_market_similarity(query_outcome_set, event)

            combined = (
                0.4 * text_sim + 0.3 * entity_sim + 0.2 * sector_sim + 0.1 * market_sim
            )

            results.append(EventSimilarityResult(
                event=event,
                similarity_score=round(combined, 4),
                text_similarity=round(text_sim, 4),
                entity_similarity=round(entity_sim, 4),
                sector_similarity=round(sector_sim, 4),
                market_similarity=round(market_sim, 4),
            ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)

        return results[:top_k]

    def find_similar_qdrant(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        self._connect()
        if not self.client:
            return self._fallback_search(query_text, top_k)

        try:
            query_vec = embedding_model.encode_query(query_text)
            search_results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vec,
                limit=top_k,
                score_threshold=score_threshold,
            )
            return [
                {"score": r.score, "event_id": r.payload.get("event_id"), **r.payload}
                for r in search_results
            ]
        except Exception:
            return self._fallback_search(query_text, top_k)

    def _fallback_search(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        results = self.find_similar(query_text, top_k=top_k)
        return [
            {
                "score": r.similarity_score,
                "event_id": r.event.id,
                "name": r.event.name,
                "description": r.event.description,
                "event_type": r.event.event_type,
            }
            for r in results
        ]

    def upsert_events(self, events: list[HistoricalEvent]):
        self._initialize()
        for ev in events:
            self._events[ev.id] = ev
        self._compute_embeddings()

        if not self.available:
            return

        try:
            from qdrant_client.http import models
            texts = [ev.name + ". " + ev.description for ev in events]
            vectors = embedding_model.encode(texts)
            qdrant_points = []
            for i, ev in enumerate(events):
                qdrant_points.append(
                    models.PointStruct(
                        id=hash(ev.id) % (2**63),
                        vector=vectors[i],
                        payload={
                            "event_id": ev.id,
                            "name": ev.name,
                            "event_type": ev.event_type,
                            "entities": ",".join(ev.entities),
                            "sectors": ",".join(ev.sectors),
                            "date": ev.date,
                        },
                    )
                )
            self.client.upsert(collection_name=self.collection, points=qdrant_points)
        except Exception:
            pass

    def aggregate_outcomes(
        self, similar_events: list[EventSimilarityResult], top_k: int = 3
    ) -> dict[str, float]:
        top = similar_events[:top_k]
        sector_impacts: dict[str, list[float]] = defaultdict(list)

        for r in top:
            for outcome in r.event.outcomes:
                sector_impacts[outcome.sector].append(outcome.impact_pct)

        aggregated = {}
        for sector, impacts in sector_impacts.items():
            aggregated[sector] = round(np.median(impacts), 1)

        return aggregated

    def build_response(
        self,
        query: str,
        similar_events: list[EventSimilarityResult],
        top_k: int = 3,
    ) -> SimilarityResponse:
        aggregated = self.aggregate_outcomes(similar_events, top_k)
        avg_score = (
            np.mean([r.similarity_score for r in similar_events[:top_k]])
            if similar_events
            else 0.0
        )
        confidence = min(0.5 + avg_score * 0.5, 0.95)

        return SimilarityResponse(
            query=query,
            similar_events=similar_events[:top_k],
            aggregated_outcomes=aggregated,
            confidence=round(confidence, 4),
        )


event_store = EventStore()


def find_similar_events(
    query_text: str,
    query_entities: Optional[list[str]] = None,
    query_sectors: Optional[list[str]] = None,
    top_k: int = 5,
) -> SimilarityResponse:
    results = event_store.find_similar(
        query_text=query_text,
        query_entities=query_entities,
        query_sectors=query_sectors,
        top_k=top_k,
    )
    return event_store.build_response(query_text, results)
