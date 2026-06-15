import json
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context
from ..knowledge.neo4j_client import Neo4jClient


class ImpactAgent:
    def __init__(self):
        self.llm = get_llm()
        self.neo4j = Neo4jClient()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=5)

        system_prompt = """You are a geopolitical impact analyst. Assess how geopolitical events affect markets, sectors, and economies.
Consider direct and indirect consequences, cascading effects, and probability-weighted outcomes."""

        prompt = f"""Query: {query}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

{'Conversation Context: ' + context.get('conversation_context', '') if context and context.get('conversation_context') else ''}

Analyze the geopolitical impact. Include:
1. Primary effects (direct consequences)
2. Secondary effects (indirect/ripple effects)
3. Sector impact analysis
4. Geographic impact scope
5. Time horizon (short/medium/long term)
6. Confidence level

Provide structured analysis:"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        entities = self._extract_entities(query)
        graph_context = ""
        for entity in entities[:3]:
            gc = self.neo4j.get_graph_context(entity)
            if gc:
                graph_context += f"\nKnowledge graph relations for {entity}:\n{gc}"

        return {
            "agent": "ImpactAgent",
            "response": response,
            "composite_risk": self._calculate_risk(response),
            "entities": entities,
            "graph_context": graph_context,
        }

    def _extract_entities(self, text: str) -> list[str]:
        prompt = f"""Extract all geopolitical entities (countries, regions, organizations, people, sectors) from this query.
Return ONLY a JSON array of strings.
Query: {text}"""
        try:
            result = self.llm.generate(prompt, temperature=0.1)
            result = result.strip().strip("```json").strip("```").strip()
            return json.loads(result) if result.startswith("[") else []
        except Exception:
            return []

    def _calculate_risk(self, text: str) -> float:
        risk_keywords = ["high", "severe", "critical", "significant", "major", "escalation",
                         "disruption", "crisis", "conflict", "war", "sanctions", "collapse"]
        count = sum(1 for kw in risk_keywords if kw in text.lower())
        return min(round(0.3 + count * 0.08, 2), 0.95)
