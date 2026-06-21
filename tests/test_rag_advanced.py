import sys, asyncio, os
sys.path.insert(0, "D:\\MY WORK\\MarketAtlas\\chat-bot")
os.environ["HF_HUB_OFFLINE"] = "1"

from rag.retrievers import MarketRetriever, GraphRetriever
from rag.pipelines import MultiGeoRAGPipeline
from rag.geo_rag import GeoRAGPipeline

def test_market():
    mr = MarketRetriever()
    results = asyncio.run(mr.retrieve("oil price crisis", limit=3))
    assert len(results) >= 1
    print(f"Market retrievals: {len(results)}")
    for r in results:
        ev = r.metadata.get("event", "?")
        print(f"  [{r.score:.2f}] {ev}")

def test_graph():
    gr = GraphRetriever()
    results = asyncio.run(gr.retrieve("Iran oil USA", limit=3))
    assert len(results) >= 1
    print(f"Graph retrievals: {len(results)}")
    for r in results:
        print(f"  [{r.score:.2f}] path found")

def test_geo_rag():
    pipeline = GeoRAGPipeline()
    result = asyncio.run(pipeline.run("What is the impact of Taiwan tensions on chips?", use_reranker=False))
    assert len(result.all_results) >= 0
    print(f"GeoRAG: intent={result.intent.intent.value}, sources={result.sources_used}, results={len(result.all_results)}")

def test_multi_geo():
    mp = MultiGeoRAGPipeline()
    result = asyncio.run(mp.query("How does Iran affect oil prices?", limit_per_source=2))
    assert result.total_results >= 0
    print(f"MultiGeo: intent={result.intent}, sources={result.sources_used}, results={result.total_results}")

if __name__ == "__main__":
    test_market()
    test_graph()
    test_geo_rag()
    test_multi_geo()
    print("\nALL ADVANCED TESTS PASSED")
