import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

from typing import Optional, Any
from ..config import settings
from .embeddings import embedding_model


class VectorStore:
    def __init__(self):
        self.client = None
        self.collection = settings.qdrant_collection

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

    def upsert(self, points: list[dict[str, Any]]):
        self._connect()
        if not self.client:
            return
        try:
            from qdrant_client.http import models
            texts = [p["text"] for p in points]
            vectors = embedding_model.encode(texts)
            qdrant_points = []
            for i, p in enumerate(points):
                qdrant_points.append(
                    models.PointStruct(
                        id=p.get("id", hash(p["text"]) % (2**63)),
                        vector=vectors[i],
                        payload={k: v for k, v in p.items() if k != "text"},
                    )
                )
            self.client.upsert(collection_name=self.collection, points=qdrant_points)
        except Exception:
            pass

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.5) -> list[dict[str, Any]]:
        self._connect()
        if not self.client:
            return []
        try:
            query_vector = embedding_model.encode_query(query)
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            return [
                {
                    "score": r.score,
                    **r.payload,
                }
                for r in results
            ]
        except Exception:
            return []


vector_store = VectorStore()

def search_knowledge(query: str, limit: int = 5) -> list[dict[str, Any]]:
    return vector_store.search(query, limit)
