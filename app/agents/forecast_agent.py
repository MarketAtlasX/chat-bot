import json
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context
from ..explain.shap_explainer import SHAPExplainer
from ..explain.attention_explainer import AttentionExplainer


class ForecastAgent:
    def __init__(self):
        self.llm = get_llm()
        self.shap = SHAPExplainer()
        self.attention = AttentionExplainer()

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

        ctx = context or {}
        ctx.update({"query": query})
        shap_result = self.shap.explain(prediction=response[:100], context=ctx)
        attn_result = self.attention.explain(prediction=response[:100], context=ctx)
        shap_formatted = self.shap.format_explanation(shap_result)

        return {
            "agent": "ForecastAgent",
            "response": response,
            "forecast_data": self._extract_forecast_data(response),
            "explanations": {
                "shap": shap_result.shap.model_dump() if shap_result.shap else None,
                "attention": attn_result.attention.model_dump() if attn_result.attention else None,
            },
            "explanation_text": shap_formatted,
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
