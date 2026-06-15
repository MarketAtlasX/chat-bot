import json
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context


class ForecastAgent:
    def __init__(self):
        self.llm = get_llm()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=4)

        system_prompt = """You are a geopolitical forecasting analyst. Generate probability-weighted forecasts
for geopolitical events and their market implications. Use historical analogies and current data."""

        prompt = f"""Query: {query}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

Generate a forecast with:
1. Most likely scenario (with probability)
2. Alternative scenarios
3. Key drivers and indicators to watch
4. Market implications for each scenario
5. Confidence level

Provide the forecast:"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        return {
            "agent": "ForecastAgent",
            "response": response,
            "forecast_data": self._extract_forecast_data(response),
        }

    def _extract_forecast_data(self, text: str) -> dict[str, Any]:
        prompt = f"""Extract forecast data from this analysis as JSON:
{text}

Return: {{"scenarios": [{{"name": str, "probability": float, "impact": str}}], "confidence": float, "timeframe": str}}
Return ONLY valid JSON."""
        try:
            result = self.llm.generate(prompt, temperature=0.1)
            result = result.strip().strip("```json").strip("```").strip()
            return json.loads(result)
        except Exception:
            return {}
