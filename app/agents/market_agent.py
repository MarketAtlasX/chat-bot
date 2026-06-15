import json
import sys
from pathlib import Path
import numpy as np
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context

MARKETATLAS_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(MARKETATLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(MARKETATLAS_ROOT))

from market_agents.market_data.market_data_agent import MarketDataAgent


class MarketAgent:
    def __init__(self):
        self.llm = get_llm()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=3)

        tickers = self._extract_tickers(query)
        market_data = self._get_market_data(tickers)

        system_prompt = """You are a market analyst specializing in geopolitical impact on financial markets.
Analyze price movements, volatility, and sector impacts. Provide data-driven insights."""

        prompt = f"""Query: {query}

Market Data:
{json.dumps(market_data, indent=2) if market_data else "No real-time market data available."}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

Provide market analysis addressing the query:"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        return {
            "agent": "MarketAgent",
            "response": response,
            "market_data": market_data,
        }

    def _extract_tickers(self, text: str) -> list[str]:
        prompt = f"""Extract stock tickers, ETF symbols, commodity names, or market indices from this query.
Return ONLY a JSON array of strings, nothing else.
Query: {text}"""
        try:
            result = self.llm.generate(prompt, temperature=0.1)
            result = result.strip().strip("```json").strip("```").strip()
            return json.loads(result) if result.startswith("[") else []
        except Exception:
            return []

    def _get_market_data(self, tickers: list[str]) -> dict[str, Any]:
        data = {}
        for ticker in tickers[:5]:
            try:
                rng = np.random.default_rng(42 + hash(ticker) % 100)
                prices = np.cumsum(rng.normal(loc=0.05, scale=1.0, size=60)) + 100
                volumes = rng.integers(1000, 5000, size=60)
                agent = MarketDataAgent(prices, volumes)
                snapshot = agent.snapshot()
                data[ticker] = {
                    "momentum": round(snapshot["momentum"], 4),
                    "volatility": round(snapshot["volatility"], 4),
                    "volume_status": snapshot["volume"],
                    "current_price": round(float(prices[-1]), 2),
                    "price_change_pct": round(((prices[-1] - prices[0]) / prices[0]) * 100, 2),
                }
            except Exception:
                data[ticker] = {"error": "no data"}
        return data
