import asyncio
import sys
sys.path.insert(0, r"D:\MY WORK\MarketAtlas\chat-bot")

from rag.chunking import TextChunker, ChunkStrategy
from rag.embeddings import BGEMModel
from rag.vectorstore import QdrantVectorStore
from rag.ingestion import NewsIngestor, Document
from rag.retrievers import NewsRetriever, MarketRetriever, HistoricalRetriever, GraphRetriever, MultiRetriever
from rag.rerankers import BGEReranker
from rag.historical_memory import EventEmbedder, EventSimilarity, AnalogRetriever
from rag.graph_retrieval import GraphDBClient, GraphQueryEngine, GraphPathExtractor
from rag.geo_rag import GeoIntentClassifier, GeoContextBuilder, GeoRAGPipeline
from rag.pipelines import (
    NewsRAGPipeline, HistoricalSimilarityPipeline, GraphRAGPipeline,
    MarketRAGPipeline, ExplainabilityRAGPipeline, MultiGeoRAGPipeline,
)


def test_chunker():
    chunker = TextChunker(chunk_size=100)
    chunks = chunker.chunk("This is a test document with multiple sentences. It should be chunked properly.")
    assert len(chunks) >= 1
    print(f"Chunker: {len(chunks)} chunks created")


def test_embeddings():
    emb = BGEMModel()
    vec = emb.embed("test query")
    assert vec.shape[0] == 1
    assert emb.dim == 1024
    print(f"Embeddings: dim={emb.dim}, available={emb.available}")


def test_intent_classifier():
    classifier = GeoIntentClassifier()

    result = classifier.classify("What historical events resemble Taiwan tensions?")
    assert result.intent.value in ["historical_analog", "multi_source"]
    print(f"Intent 1: {result.intent.value}, conf={result.confidence:.2f}, entities={result.extracted_entities}")

    result2 = classifier.classify("How does Iran affect European energy?")
    print(f"Intent 2: {result2.intent.value}, conf={result2.confidence:.2f}, entities={result2.extracted_entities}")

    result3 = classifier.classify("What happened to oil during similar crises?")
    print(f"Intent 3: {result3.intent.value}, conf={result3.confidence:.2f}")

    result4 = classifier.classify("Will Taiwan tensions affect Nvidia?")
    print(f"Intent 4: {result4.intent.value}, conf={result4.confidence:.2f}, sectors={result4.extracted_sectors}")


def test_historical_events():
    hist = HistoricalRetriever()
    events = hist.search_events("oil energy middle east", limit=3)
    assert len(events) > 0
    print(f"Historical: {len(events)} matching events")
    for e in events:
        print(f"  - {e['name']}")


def test_event_similarity():
    sim = EventSimilarity()
    similar = sim.find_similar("Oil supply disruption in Middle East", top_k=3)
    assert len(similar) > 0
    print(f"Similar events: {len(similar)}")
    for s in similar:
        print(f"  - {s.event_name}: {s.combined_score:.2f}")


def test_graph_retriever():
    gr = GraphRetriever()
    results = asyncio.run(gr.retrieve("How does Iran affect oil?", limit=3))
    assert len(results) > 0
    print(f"Graph retriever: {len(results)} results")
    for r in results:
        preview = r.content[:80]
        print(f"  - [{r.score:.2f}] {preview}...")


def test_analog_retriever():
    ar = AnalogRetriever()
    analogs = ar.find_analogs("Taiwan tensions and semiconductor supply chain", top_k=3)
    assert len(analogs.analogs) > 0
    print(f"Analogs: {len(analogs.analogs)}")
    for a in analogs.analogs:
        print(f"  - {a.event_name}: {a.combined_score:.2f}")


def test_graph_paths():
    gpe = GraphPathExtractor()
    paths = gpe.extract_paths("How does Iran affect European energy?")
    assert len(paths) > 0
    print(f"Graph paths: {len(paths)}")
    for p in paths[:3]:
        print(f"  - {p.path_string[:80]}...")


async def test_geo_pipeline():
    pipeline = GeoRAGPipeline()
    result = await pipeline.run("What are the implications of Taiwan tensions on global trade?")
    assert result.prompt is not None
    assert len(result.sources_used) > 0
    print(f"GeoRAG: intent={result.intent.intent.value}, sources={result.sources_used}, results={len(result.all_results)}")


def test_market_retriever():
    mr = MarketRetriever()
    results = asyncio.run(mr.retrieve("What happened to oil during geopolitical crises?", limit=3))
    assert len(results) > 0
    print(f"Market retriever: {len(results)} results")
    for r in results:
        print(f"  - [{r.score:.2f}] {r.metadata.get('event', 'unknown')}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RAG Pipeline Modules")
    print("=" * 60)

    test_chunker()
    test_embeddings()
    test_intent_classifier()
    test_historical_events()
    test_event_similarity()
    test_graph_retriever()
    test_analog_retriever()
    test_graph_paths()
    test_market_retriever()
    asyncio.run(test_geo_pipeline())

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
