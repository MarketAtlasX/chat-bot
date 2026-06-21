from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rag.chunking import TextChunker, ChunkResult
from rag.embeddings import get_embedding_model
from rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    source: str = "unknown"
    title: str = ""
    url: str = ""
    published_at: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestResult:
    document_id: str
    chunk_ids: List[str] = field(default_factory=list)
    num_chunks: int = 0
    success: bool = False
    error: Optional[str] = None


class NewsIngestor:
    def __init__(self, collection: str = "marketatlas_news"):
        self.collection = collection
        self.embedder = get_embedding_model()
        self.chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        self.store = get_vector_store(collection)

    def ingest(self, document: Document) -> IngestResult:
        try:
            chunks = self.chunker.chunk(
                document.text,
                metadata={
                    "source": document.source,
                    "title": document.title,
                    "url": document.url,
                    "published_at": document.published_at or datetime.utcnow().isoformat(),
                    "topics": ",".join(document.topics),
                    "sentiment": str(document.sentiment) if document.sentiment else "",
                    **document.metadata,
                },
            )
            texts = [c.text for c in chunks]
            vectors = self.embedder.embed(texts)
            metadatas = [
                {
                    "chunk_index": c.index,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "source": document.source,
                    "title": document.title,
                    "url": document.url,
                    "published_at": document.published_at or "",
                    "topics": ",".join(document.topics),
                    "sentiment": str(document.sentiment) if document.sentiment else "",
                    **document.metadata,
                }
                for c in chunks
            ]
            chunk_ids = self.store.add(texts, vectors, metadatas)
            return IngestResult(
                document_id=chunk_ids[0] if chunk_ids else "",
                chunk_ids=chunk_ids,
                num_chunks=len(chunks),
                success=True,
            )
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return IngestResult(
                document_id="",
                success=False,
                error=str(e),
            )

    def ingest_batch(self, documents: List[Document]) -> List[IngestResult]:
        return [self.ingest(doc) for doc in documents]
