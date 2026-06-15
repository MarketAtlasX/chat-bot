import json
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context


class RecommendationAgent:
    def __init__(self):
        self.llm = get_llm()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=5)

        system_prompt = """You are an investment strategist specializing in geopolitical trading.
Provide actionable investment recommendations based on geopolitical analysis.
Consider: sectors, specific assets, risk levels, time horizons, and portfolio positioning."""

        prompt = f"""Query: {query}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

{'Market Context: ' + json.dumps(context.get('market_data', {})) if context and context.get('market_data') else ''}
{'Impact Analysis: ' + context.get('impact_analysis', '') if context and context.get('impact_analysis') else ''}

Provide investment recommendations including:
1. Recommended assets (with tickers/ETFs)
2. Direction (buy/sell/hold)
3. Conviction level
4. Time horizon
5. Risk assessment
6. Hedging suggestions

Provide the analysis:"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        return {
            "agent": "RecommendationAgent",
            "response": response,
            "recommendations": self._extract_recommendations(response),
        }

    def _extract_recommendations(self, text: str) -> list[dict]:
        prompt = f"""Extract investment recommendations from this text as a JSON array:
{text}

Return: [{{"asset": str, "action": "BUY"|"SELL"|"HOLD", "conviction": float (0-1), "reasoning": str}}]
Return ONLY valid JSON array."""
        try:
            result = self.llm.generate(prompt, temperature=0.1)
            result = result.strip().strip("```json").strip("```").strip()
            return json.loads(result) if result.startswith("[") else []
        except Exception:
            return []
