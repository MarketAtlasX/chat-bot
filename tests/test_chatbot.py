import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from app.agents.intent_router import IntentRouter
from app.agents.news_agent import NewsAgent
from app.agents.impact_agent import ImpactAgent
from app.agents.market_agent import MarketAgent
from app.agents.graph_agent import GraphAgent
from app.agents.simulation_agent import SimulationAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.agents.event_similarity_agent import EventSimilarityAgent
from app.models import IntentType
from app.rag.retriever import seed_knowledge_base
from app.event_memory.event_store import event_store


class TestIntentRouter:
    def setup_method(self):
        self.router = IntentRouter()

    def test_news_intent(self):
        intent, conf = self.router.classify("Show me the latest news about Iran sanctions")
        assert intent == IntentType.NEWS
        assert conf > 0.5

    def test_market_intent(self):
        intent, conf = self.router.classify("Why is oil rising today?")
        assert intent == IntentType.IMPACT or intent == IntentType.MARKET

    def test_recommendation_intent(self):
        intent, conf = self.router.classify("Should I buy defense stocks?")
        assert intent == IntentType.RECOMMENDATION
        assert conf > 0.5

    def test_simulation_intent(self):
        intent, conf = self.router.classify("What if Russia cuts gas exports by 30%?")
        assert intent == IntentType.SIMULATION
        assert conf > 0.5

    def test_graph_intent(self):
        intent, conf = self.router.classify("How is Russia related to Europe energy?")
        assert intent == IntentType.GRAPH
        assert conf > 0.5

    def test_report_intent(self):
        intent, conf = self.router.classify("Generate an intelligence report on Taiwan")
        assert intent == IntentType.REPORT
        assert conf > 0.5

    def test_similarity_intent(self):
        intent, conf = self.router.classify("What historical events are similar to the current Iran-Israel tensions?")
        assert intent == IntentType.SIMILARITY
        assert conf > 0.5

    def test_similarity_intent_alt(self):
        intent, conf = self.router.classify("Find past events like the Russia-Ukraine war")
        assert intent == IntentType.SIMILARITY
        assert conf > 0.5


class TestEventSimilarityAgent:
    def setup_method(self):
        self.agent = EventSimilarityAgent()

    def test_process_returns_formatted_response(self):
        result = self.agent.process("Iran-Israel tensions in the Middle East affecting oil markets")
        assert result["agent"] == "EventSimilarityAgent"
        assert "response" in result
        assert len(result["response"]) > 0
        assert "Historical Event Similarity" in result["response"]

    def test_similarity_engine_finds_matching_events(self):
        results = event_store.find_similar(
            query_text="Iran-Israel military conflict in the Middle East",
            query_entities=["Iran", "Israel", "Middle East"],
            query_sectors=["Energy", "Defense"],
            top_k=3,
        )
        assert len(results) > 0
        for r in results:
            assert 0 <= r.similarity_score <= 1
            assert r.event.name

    def test_aggregated_outcomes(self):
        results = event_store.find_similar(
            query_text="Oil supply disruption in the Middle East due to conflict",
            top_k=3,
        )
        response = event_store.build_response("test", results)
        assert len(response.similar_events) > 0
        if response.aggregated_outcomes:
            for sector, impact in response.aggregated_outcomes.items():
                assert isinstance(impact, (int, float))


class TestNewsAgent:
    def setup_method(self):
        self.agent = NewsAgent()

    def test_process_returns_response(self):
        result = self.agent.process("What happened with oil prices?")
        assert "agent" in result
        assert "response" in result
        assert result["agent"] == "NewsAgent"
        assert len(result["response"]) > 0


class TestImpactAgent:
    def setup_method(self):
        self.agent = ImpactAgent()

    def test_process_returns_risk(self):
        result = self.agent.process("Iran tensions affecting oil markets")
        assert "agent" in result
        assert "response" in result
        assert "composite_risk" in result
        assert 0 <= result["composite_risk"] <= 1


class TestMarketAgent:
    def setup_method(self):
        self.agent = MarketAgent()

    def test_process_returns_market_data(self):
        result = self.agent.process("How is the energy sector performing?")
        assert "response" in result
        assert result["agent"] == "MarketAgent"


class TestGraphAgent:
    def setup_method(self):
        self.agent = GraphAgent()

    def test_process_returns_entities(self):
        result = self.agent.process("How are Russia, gas, and Europe connected?")
        assert "response" in result
        assert "entities" in result


class TestSimulationAgent:
    def setup_method(self):
        self.agent = SimulationAgent()

    def test_process_returns_simulation(self):
        result = self.agent.process("What if the US imposes more sanctions on China?")
        assert "response" in result
        assert "simulation_data" in result


class TestRecommendationAgent:
    def setup_method(self):
        self.agent = RecommendationAgent()

    def test_process_returns_recommendations(self):
        result = self.agent.process("What stocks benefit from a Taiwan blockade?")
        assert "response" in result
        assert "recommendations" in result


class TestReportAgent:
    def setup_method(self):
        self.agent = ReportAgent()

    def test_process_returns_report(self):
        result = self.agent.process("Generate report on Iranian naval activity")
        assert "response" in result
        assert "report_data" in result or len(result["response"]) > 0


class TestKnowledgeBase:
    def test_seed_knowledge_base(self):
        result = seed_knowledge_base()
        assert result is True or result is False


if __name__ == "__main__":
    pytest.main(["-v", __file__])
