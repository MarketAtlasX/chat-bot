from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: str
    score: float
    payload: dict = field(default_factory=dict)
    vector: Optional[List[float]] = None


class QdrantVectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "marketatlas",
        vector_dim: int = 1024,
    ):
        self.url = url
        self.collection = collection
        self.vector_dim = vector_dim
        self._client = None
        self._available = False
        self._connect()

    def _connect(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            self._models = models
            self._client = QdrantClient(url=self.url, timeout=5)
            self._client.get_collections()
            self._ensure_collection()
            self._available = True
            logger.info(f"Connected to Qdrant at {self.url}")
        except Exception as e:
            self._available = False
            self._client = None
            logger.warning(f"Qdrant unavailable at {self.url}: {e}")

    def _ensure_collection(self):
        collections = self._client.get_collections().collections
        existing = [c.name for c in collections]
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=self._models.VectorParams(
                    size=self.vector_dim,
                    distance=self._models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    @property
    def available(self) -> bool:
        return self._available

    def add(
        self,
        texts: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not self._available or self._client is None:
            return ids or [str(uuid4()) for _ in texts]
        try:
            point_ids = ids or [str(uuid4()) for _ in texts]
            metadatas = metadata or [{}] * len(texts)
            points = []
            for i, (text, vec) in enumerate(zip(texts, vectors)):
                points.append(
                    self._models.PointStruct(
                        id=point_ids[i],
                        vector=vec.tolist(),
                        payload={"text": text, **metadatas[i]},
                    )
                )
            self._client.upsert(collection_name=self.collection, points=points, wait=True)
            return point_ids
        except Exception as e:
            logger.error(f"Qdrant add failed: {e}")
            self._available = False
            return ids or [str(uuid4()) for _ in texts]

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[dict] = None,
    ) -> List[SearchResult]:
        if not self._available or self._client is None:
            return []
        try:
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    must_conditions.append(
                        self._models.FieldCondition(
                            key=key,
                            match=self._models.MatchValue(value=value),
                        )
                    )
                query_filter = self._models.Filter(must=must_conditions)
            if hasattr(self._client, "query_points"):
                results = self._client.query_points(
                    collection_name=self.collection,
                    query=query_vector.tolist(),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                ).points
            else:
                results = self._client.search(
                    collection_name=self.collection,
                    query_vector=query_vector.tolist(),
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                )
            return [
                SearchResult(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                    vector=r.vector,
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    def delete(self, ids: List[str]):
        if not self._available:
            return
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=self._models.PointIdsList(
                    points=[str(i) for i in ids]
                ),
            )
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")

    def list_collections(self) -> List[str]:
        if not self._available:
            return []
        try:
            return [c.name for c in self._client.get_collections().collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []


_store_instance: Optional[Tuple[str, QdrantVectorStore]] = None


def get_vector_store(collection: str = "marketatlas") -> QdrantVectorStore:
    global _store_instance
    if _store_instance is None or _store_instance[0] != collection:
        _store_instance = (collection, QdrantVectorStore(collection=collection))
    return _store_instance[1]
