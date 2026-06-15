import httpx
import json
import random
from typing import Optional, Generator, List
from .base import LLMInterface
from ..config import settings


class MockLLM(LLMInterface):
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        query = prompt.split("Query: ")[-1].split("\n")[0].strip() if "Query: " in prompt else prompt[:100]

        if "classify" in prompt.lower() or "category" in prompt.lower():
            categories = ["NEWS", "MARKET", "IMPACT", "RECOMMENDATION", "SIMULATION", "GRAPH", "REPORT"]
            keywords = {
                "oil": "IMPACT", "sanction": "NEWS", "buy": "RECOMMENDATION", "sell": "RECOMMENDATION",
                "invest": "RECOMMENDATION", "simulate": "SIMULATION", "what if": "SIMULATION",
                "scenario": "SIMULATION", "relationship": "GRAPH", "connection": "GRAPH",
                "report": "REPORT", "intelligence": "REPORT", "news": "NEWS", "latest": "NEWS",
                "price": "MARKET", "market": "MARKET", "stock": "MARKET",
            }
            q = query.lower()
            for kw, cat in keywords.items():
                if kw in q:
                    return cat
            return random.choice(categories)

        if "Extract" in prompt or "JSON" in prompt:
            if "Extract geopolitical entities" in prompt or "Extract all named entities" in prompt:
                entities = []
                for word in query.split():
                    if word[0].isupper() and len(word) > 2:
                        entities.append(word.strip(".,;:!?"))
                if not entities:
                    entities = ["Iran", "Oil", "Energy"] if "oil" in query.lower() else ["Russia", "Europe", "Gas"]
                return json.dumps(list(set(entities[:5])))
            if "stock tickers" in prompt:
                tickers = []
                for word in query.split():
                    w = word.strip(".,;:!?").upper()
                    if len(w) <= 5 and w.isalpha() and w != word:
                        tickers.append(w)
                if not tickers:
                    tickers = ["XLE", "CVX", "XOM"] if "energy" in query.lower() or "oil" in query.lower() else ["SPY", "QQQ"]
                return json.dumps(tickers[:5])

        return self._mock_response(query, system_prompt, prompt)

    def _mock_response(self, query: str, system_prompt: Optional[str] = None, full_prompt: str = "") -> str:
        if "simulate" in query.lower() or "what if" in query.lower() or "scenario" in query.lower():
            return json.dumps({
                "scenario": f"Scenario: {query}",
                "consequences": {"Oil": "+12%", "European Manufacturing": "-5%", "Inflation": "+2.3%"},
                "probability": 0.71,
                "time_horizon": "medium term",
                "key_risks": ["Supply chain disruption", "Inflation spike", "Market volatility"]
            })

        oil_keywords = ["oil", "energy", "crude", "gas", "petroleum"]
        defense_keywords = ["defense", "military", "lockheed", "northrop", "weapon"]
        sanction_keywords = ["sanction", "tariff", "trade war", "embargo"]
        conflict_keywords = ["conflict", "war", "attack", "tension", "strike", "blockade"]
        market_keywords = ["market", "stock", "price", "etf", "index", "rally", "decline"]

        if any(k in query.lower() for k in oil_keywords):
            return "Analysis indicates energy markets are experiencing upward pressure due to geopolitical tensions. Primary drivers include supply constraints, rising shipping costs, and increased risk premium. The energy sector shows strong momentum with elevated volatility suggesting continued uncertainty."
        if any(k in query.lower() for k in defense_keywords):
            return "Defense sector analysis shows increased demand expectations driven by rising geopolitical tensions. Key beneficiaries include major defense contractors. Safe-haven flows also support gold and energy ETFs."
        if any(k in query.lower() for k in sanction_keywords):
            return "Sanctions analysis indicates significant market disruption potential. Affected sectors include energy, finance, and shipping. Supply chain reconfiguration expected with medium-term inflationary pressure."
        if any(k in query.lower() for k in conflict_keywords):
            return "Geopolitical risk assessment shows elevated tension levels. Direct market impacts expected in energy, defense, and safe-haven assets. Confidence level: moderate to high based on current intelligence indicators."
        if any(k in query.lower() for k in market_keywords):
            return "Market analysis indicates mixed signals. Momentum shows positive trend but volatility remains elevated. Volume patterns suggest institutional positioning. Recommend monitoring key support levels."
        if "report" in query.lower() or "intelligence" in query.lower():
            return json.dumps({
                "title": "Geopolitical Intelligence Report",
                "event": f"Analysis of: {query}",
                "affected_sectors": ["Energy", "Defense", "Financials"],
                "risk_score": 0.72,
                "expected_market_impact": "Moderate to significant impact expected across affected sectors",
                "recommended_assets": ["XLE", "GDX", "TLT"],
                "confidence": 0.78,
                "reasoning": "Based on current geopolitical indicators and market positioning analysis",
                "sources": ["MarketAtlas Intelligence", "Reuters", "Bloomberg"]
            })

        return f"Analysis complete for: {query[:100]}... Assessment based on available intelligence indicates moderate geopolitical risk with potential market implications across affected sectors."

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> Generator[str, None, None]:
        result = self.generate(prompt, system_prompt, temperature)
        for chunk in result.split(". "):
            yield chunk + ". "


class OllamaLLM(LLMInterface):
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or settings.llm_model
        self.base_url = base_url or settings.ollama_base_url
        self._available = None

    def _check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            with httpx.Client(base_url=self.base_url, timeout=2) as client:
                resp = client.get("/api/tags")
                self._available = resp.status_code == 200
                return self._available
        except Exception:
            self._available = False
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        if not self._check_available():
            mock = MockLLM()
            return mock.generate(prompt, system_prompt, temperature)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(base_url=self.base_url, timeout=30) as client:
                resp = client.post("/api/generate", json=payload)
                resp.raise_for_status()
                return resp.json().get("response", "")
        except Exception as e:
            mock = MockLLM()
            return mock.generate(prompt, system_prompt, temperature)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> Generator[str, None, None]:
        if not self._check_available():
            mock = MockLLM()
            yield from mock.generate_stream(prompt, system_prompt, temperature)
            return

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with httpx.Client(base_url=self.base_url, timeout=30) as client:
                with client.stream("POST", "/api/generate", json=payload) as resp:
                    for line in resp.iter_lines():
                        if line:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
        except Exception:
            mock = MockLLM()
            yield from mock.generate_stream(prompt, system_prompt, temperature)


def get_llm() -> LLMInterface:
    return OllamaLLM()
