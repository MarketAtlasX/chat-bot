from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.historical_memory.analog_retriever import AnalogRetriever
from rag.graph_retrieval.graph_paths import GraphPathExtractor
from rag.pipelines.historical_similarity import HistoricalSimilarityPipeline
from rag.pipelines.graph_rag import GraphRAGPipeline
from rag.pipelines.market_rag import MarketRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityRAGResult:
    query: str
    prediction_context: str = ""
    historical_analogs: List[dict] = field(default_factory=list)
    graph_paths: List[dict] = field(default_factory=list)
    market_reactions: List[dict] = field(default_factory=list)
    reasoning_factors: List[dict] = field(default_factory=list)
    explanation: str = ""


EXPLANATION_TEMPLATES = {
    "energy": {
        "factors": [
            "Oil supply disruption risk",
            "Geopolitical premium in pricing",
            "Historical analog similarity to past energy crises",
            "Supply chain chokepoint vulnerability",
            "Sector momentum and positioning",
        ],
        "template": "The prediction is driven by {factor_count} key factors. Primary among them is {primary}. "
                    "Historical analogs show {analog_count} similar past events with comparable market reactions.",
    },
    "technology": {
        "factors": [
            "Supply chain concentration risk",
            "Geopolitical tension impact on semiconductor access",
            "Historical analog similarity to past tech disruptions",
            "Market pricing of geopolitical risk",
            "Sector valuation relative to historical norms",
        ],
        "template": "Technology sector exposure is driven by {factor_count} factors. "
                    "The most significant is {primary}. "
                    "{analog_count} historical analogs support this assessment.",
    },
    "defense": {
        "factors": [
            "Geopolitical escalation probability",
            "Defense spending cycle positioning",
            "Historical analog similarity to past conflicts",
            "Government budget allocation trends",
            "Supply chain and procurement risks",
        ],
        "template": "Defense sector outlook is shaped by {factor_count} factors. "
                    "Key driver is {primary}. "
                    "Historical context from {analog_count} similar situations reinforces this analysis.",
    },
    "finance": {
        "factors": [
            "Interest rate sensitivity",
            "Credit risk from geopolitical exposure",
            "Market volatility regime",
            "Capital flow patterns",
            "Historical analog similarity to past financial crises",
        ],
        "template": "Financial sector assessment considers {factor_count} factors. "
                    "Primary among them is {primary}. "
                    "{analog_count} historical analogs inform the risk assessment.",
    },
    "general": {
        "factors": [
            "Geopolitical risk assessment",
            "Historical analog similarity",
            "Graph relationship analysis",
            "Market reaction patterns",
            "Sector-specific dynamics",
        ],
        "template": "Analysis based on {factor_count} factors. "
                    "Primary driver: {primary}. "
                    "Supported by {analog_count} historical analogs and graph relationship analysis.",
    },
}


class ExplainabilityRAGPipeline:
    def __init__(self):
        self.historical_pipeline = HistoricalSimilarityPipeline()
        self.graph_pipeline = GraphRAGPipeline()
        self.market_pipeline = MarketRAGPipeline()
        self.analog_retriever = AnalogRetriever()
        self.path_extractor = GraphPathExtractor()

    async def explain(
        self,
        prediction: str,
        prediction_type: str = "general",
    ) -> ExplainabilityRAGResult:
        analogs_result = self.historical_pipeline.find_analogs(prediction, top_k=3)
        graph_paths = self.path_extractor.extract_paths(prediction, max_paths=5)
        market_result = await self.market_pipeline.query(prediction, limit=3)

        template_key = prediction_type if prediction_type in EXPLANATION_TEMPLATES else "general"
        template = EXPLANATION_TEMPLATES[template_key]
        factors = template["factors"]
        reasoning_factors = [
            {"factor": f, "source": self._get_factor_source(f, analogs_result, graph_paths, market_result)}
            for f in factors
        ]
        analog_count = len(analogs_result.analogs)
        primary = factors[0] if factors else "geopolitical dynamics"
        explanation_template = template["template"].format(
            factor_count=len(factors),
            primary=primary,
            analog_count=analog_count,
        )
        extra_lines = []
        if analogs_result.analogs:
            extra_lines.append(f"\n\nHistorical Analogs:")
            for a in analogs_result.analogs[:3]:
                extra_lines.append(f"  - {a['event_name']} (similarity: {a['score']:.2f}): {a['description'][:100]}")
        if graph_paths:
            extra_lines.append(f"\n\nGraph Relationships:")
            for p in graph_paths[:3]:
                extra_lines.append(f"  - {p.path_string}")
        if market_result.reactions:
            extra_lines.append(f"\n\nMarket Reactions:")
            for r in market_result.reactions[:3]:
                extra_lines.append(f"  - {r['event']}: {r['reaction'][:100]}")
        explanation = explanation_template + "\n".join(extra_lines)
        return ExplainabilityRAGResult(
            query=prediction,
            prediction_context=f"Prediction: {prediction} (type: {prediction_type})",
            historical_analogs=[
                {"event_name": a["event_name"], "score": a["score"], "description": a.get("description", "")}
                for a in analogs_result.analogs
            ],
            graph_paths=[
                {"path": gp.path_string, "entities": gp.entities, "sectors": gp.sectors}
                for gp in graph_paths
            ],
            market_reactions=market_result.reactions,
            reasoning_factors=reasoning_factors,
            explanation=explanation,
        )

    def _get_factor_source(
        self,
        factor: str,
        analogs_result: Any,
        graph_paths: Any,
        market_result: Any,
    ) -> str:
        if "supply" in factor.lower() or "disruption" in factor.lower():
            return "market_reactions" if market_result.reactions else "graph_paths"
        if "historical" in factor.lower():
            return "historical_analogs"
        if "graph" in factor.lower() or "relationship" in factor.lower() or "chain" in factor.lower():
            return "graph_paths"
        if "momentum" in factor.lower() or "valuation" in factor.lower() or "pricing" in factor.lower():
            return "market_reactions"
        return "geopolitical_analysis"

    async def query(self, question: str, **kwargs) -> ExplainabilityRAGResult:
        return await self.explain(question, **kwargs)
