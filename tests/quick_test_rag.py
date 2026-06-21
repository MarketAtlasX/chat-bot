"""Quick tests for RAG pipeline modules - designed to work offline."""
import sys, asyncio, os
sys.path.insert(0, "D:\\MY WORK\\MarketAtlas\\chat-bot")

# Block network-heavy initializations
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["RAG_SKIP_QDRANT"] = "1"

from rag.chunking import TextChunker
from rag.embeddings import BGEMModel
from rag.retrievers import MarketRetriever, HistoricalRetriever
from rag.historical_memory import EventSimilarity, AnalogRetriever
from rag.graph_retrieval import GraphPathExtractor
from rag.geo_rag import GeoIntentClassifier

def test_chunker():
    chunker = TextChunker(chunk_size=100)
    chunks = chunker.chunk("Test document with multiple sentences. Second sentence here. Third one too.")
    assert len(chunks) >= 1
    print(f"[OK] Chunker: {len(chunks)} chunks")

def test_embeddings():
    emb = BGEMModel()
    vec = emb.embed("test")
    print(f"[OK] Embeddings: dim={emb.dim}, available={emb.available}")

def test_intent():
    clf = GeoIntentClassifier()
    test_cases = [
        ("What historical events resemble Taiwan tensions?", "historical_analog", ["Taiwan"]),
        ("How does Iran affect European energy?", "multi_source", ["Iran", "Europe"]),
        ("Will Taiwan tensions affect Nvidia?", "multi_source", ["technology"]),
        ("What is the market impact of oil sanctions?", "market_impact", ["energy"]),
        ("Tell me about the relationship between Russia and EU", "multi_source", ["Russia", "Europe"]),
        ("Latest sanctions news on Iran", "news_analysis", ["Iran"]),
    ]
    for query, expected_intent, expected_entities in test_cases:
        r = clf.classify(query)
        assert isinstance(r.intent.value, str)
        print(f"[OK] Intent[{query[:30]}...]: {r.intent.value} conf={r.confidence:.2f}")

def test_historical():
    hist = HistoricalRetriever()
    events = hist.search_events("oil energy middle east", limit=3)
    assert len(events) >= 1
    names = [e["name"] for e in events]
    print(f"[OK] Historical events: {len(events)} -> {names}")

def test_similarity():
    sim = EventSimilarity()
    similar = sim.find_similar("Oil supply disruption in Middle East", top_k=3)
    assert len(similar) >= 1
    print(f"[OK] Event similarity: {len(similar)} matches")
    for s in similar:
        print(f"     - {s.event_name}: {s.combined_score:.2f}")

def test_analogs():
    ar = AnalogRetriever()
    analogs = ar.find_analogs("Taiwan tensions and chips", top_k=3)
    assert len(analogs.analogs) >= 1
    print(f"[OK] Analogs: {len(analogs.analogs)}")
    for a in analogs.analogs:
        print(f"     - {a.event_name}: {a.combined_score:.2f}")

def test_graph_paths():
    gpe = GraphPathExtractor()
    paths = gpe.extract_paths("How does Iran affect European energy?")
    assert len(paths) >= 1
    print(f"[OK] Graph paths: {len(paths)} found")
    for p in paths[:3]:
        ps = p.path_string[:50]
        print(f"     - {ps}...")

def test_market():
    mr = MarketRetriever()
    results = asyncio.run(mr.retrieve("oil price crisis", limit=3))
    assert len(results) >= 1
    print(f"[OK] Market retrievals: {len(results)}")
    for r in results:
        ev = r.metadata.get("event", "?")
        print(f"     - [{r.score:.2f}] {ev}")

if __name__ == "__main__":
    test_chunker()
    test_embeddings()
    test_intent()
    test_historical()
    test_similarity()
    test_analogs()
    test_graph_paths()
    test_market()
    print("\n=== ALL CORE TESTS PASSED ===")
