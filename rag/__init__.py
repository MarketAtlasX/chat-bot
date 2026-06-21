from rag.chunking import TextChunker, ChunkStrategy, ChunkResult
from rag.embeddings import BGEMModel, get_embedding_model
from rag.vectorstore import QdrantVectorStore, SearchResult, get_vector_store
from rag.ingestion import NewsIngestor, Document, IngestResult
from rag.retrievers import (
    BaseRetriever,
    RetrievalResult,
    RetrieverType,
    NewsRetriever,
    MarketRetriever,
    HistoricalRetriever,
    GraphRetriever,
    MultiRetriever,
    MultiRetrievalResult,
)
from rag.rerankers import BGEReranker, RerankResult, get_reranker
from rag.historical_memory import EventEmbedder, EventSimilarity, SimilarityScore, AnalogRetriever, AnalogResult
from rag.graph_retrieval import GraphDBClient, GraphQueryEngine, GraphQueryResult, GraphPathExtractor, GraphPath
from rag.geo_rag import GeoIntentClassifier, GeoIntent, IntentResult, GeoContextBuilder, GeoRAGPipeline, GeoRAGResult
from rag.embeddings import get_embedding_model
from rag.ingestion import Document, NewsIngestor
from rag.pipelines import (
    NewsRAGPipeline,
    HistoricalSimilarityPipeline,
    GraphRAGPipeline,
    MarketRAGPipeline,
    ExplainabilityRAGPipeline,
    MultiGeoRAGPipeline,
)


def seed_knowledge_base():
    ingestor = NewsIngestor(collection="marketatlas_news")
    docs = [
        Document(
            text="Oil prices surged 25% in 3 months after US withdrawal from Iran nuclear deal.",
            source="seed", title="Iran Sanctions Impact",
            topics=["energy", "sanctions"], sentiment=-0.3,
        ),
        Document(
            text="European gas prices increased 400% following the Russian invasion of Ukraine.",
            source="seed", title="Russia Ukraine Gas Crisis",
            topics=["energy", "conflict"], sentiment=-0.5,
        ),
        Document(
            text="Taiwan semiconductor supply chain disruption risk after Chinese military exercises.",
            source="seed", title="Taiwan Chip Risk",
            topics=["technology", "conflict"], sentiment=-0.4,
        ),
    ]
    results = ingestor.ingest_batch(docs)
    return sum(1 for r in results if r.success)


__all__ = [
    "TextChunker", "ChunkStrategy", "ChunkResult",
    "BGEMModel", "get_embedding_model",
    "QdrantVectorStore", "SearchResult", "get_vector_store",
    "NewsIngestor", "Document", "IngestResult",
    "BaseRetriever", "RetrievalResult", "RetrieverType",
    "NewsRetriever", "MarketRetriever", "HistoricalRetriever", "GraphRetriever",
    "MultiRetriever", "MultiRetrievalResult",
    "BGEReranker", "RerankResult", "get_reranker",
    "EventEmbedder", "EventSimilarity", "SimilarityScore", "AnalogRetriever", "AnalogResult",
    "GraphDBClient", "GraphQueryEngine", "GraphQueryResult", "GraphPathExtractor", "GraphPath",
    "GeoIntentClassifier", "GeoIntent", "IntentResult", "GeoContextBuilder", "GeoRAGPipeline", "GeoRAGResult",
    "NewsRAGPipeline", "HistoricalSimilarityPipeline", "GraphRAGPipeline",
    "MarketRAGPipeline", "ExplainabilityRAGPipeline", "MultiGeoRAGPipeline",
]
