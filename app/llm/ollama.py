import httpx
import json
from typing import Optional, Generator
from .base import LLMInterface
from ..config import settings


KNOWN_ENTITIES_MAP = {
    "iran": "Iran", "israel": "Israel", "russia": "Russia", "ukraine": "Ukraine",
    "china": "China", "taiwan": "Taiwan", "us": "United States", "usa": "United States",
    "middle east": "Middle East", "europe": "Europe", "nato": "NATO",
    "uk": "United Kingdom", "britain": "United Kingdom", "germany": "Germany",
    "france": "France", "india": "India", "japan": "Japan", "korea": "South Korea",
    "saudi": "Saudi Arabia", "uae": "UAE", "turkey": "Turkey",
    "africa": "Africa", "asia": "Asia", "pacific": "Asia Pacific",
    "ruble": "Russia", "yuan": "China", "yen": "Japan",
    "opec": "OPEC", "un": "United Nations", "eu": "European Union",
    "houthi": "Houthi Rebels", "hezbollah": "Hezbollah",
    "hamas": "Hamas", "taliban": "Taliban", "isis": "ISIS",
    "putin": "Russia", "xi": "China", "biden": "United States",
    "trump": "United States", "netanyahu": "Israel",
    "tsmc": "Taiwan", "aramco": "Saudi Arabia",
    "strait of hormuz": "Strait of Hormuz", "suez": "Suez Canal",
    "red sea": "Red Sea", "gulf": "Persian Gulf",
    "wall street": "United States",
}

SECTOR_KEYWORDS = {
    "energy": ["oil", "gas", "energy", "petroleum", "crude", "fuel", "brent", "wti", "opec", "xle"],
    "defense": ["defense", "military", "weapon", "army", "navy", "security", "defence", "lockheed", "northrop", "ita"],
    "technology": ["tech", "technology", "semiconductor", "chip", "software", "cyber", "ai", "artificial intelligence", "xlk", "qqq"],
    "financials": ["bank", "finance", "financial", "insurance", "interest rate", "fed", "federal reserve", "xlf"],
    "shipping": ["shipping", "port", "maritime", "trade route", "supply chain", "container", "freight"],
    "agriculture": ["agriculture", "wheat", "grain", "food", "farm", "corn", "soybean"],
    "airlines": ["airline", "aviation", "flight", "travel", "airport", "jet"],
    "manufacturing": ["manufacturing", "factory", "industrial", "production", "automotive"],
    "healthcare": ["health", "medical", "pharma", "hospital", "xlv"],
    "cybersecurity": ["cybersecurity", "ransomware", "hack", "breach", "cyber attack"],
    "precious metals": ["gold", "silver", "gld", "platinum", "safe haven"],
}


class MockLLM(LLMInterface):
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.3) -> str:
        query = self._extract_query(prompt)
        prompt_lower = prompt.lower()

        if "classify" in prompt_lower or "category" in prompt_lower:
            return self._classify(query)

        if "extract" in prompt_lower or "json array" in prompt_lower or "json" in prompt_lower:
            if "entity" in prompt_lower or "geopolitical" in prompt_lower or "named" in prompt_lower:
                return self._extract_entities_json(query)
            if "sector" in prompt_lower:
                return self._extract_sectors_json(query)
            if "stock ticker" in prompt_lower or "ticker" in prompt_lower:
                return self._extract_tickers_json(query)
            if "forecast" in prompt_lower or "scenario" in prompt_lower:
                return self._generate_forecast_json(query)
            if "report" in prompt_lower:
                return self._generate_report_json(query)

        if "return only valid json" in prompt_lower or "return only the json" in prompt_lower:
            return self._handle_json_prompt(query, prompt)

        if "intelligence report" in prompt_lower or "marketatlas" in prompt_lower:
            return self._generate_report_json(query)

        return self._mock_response(query)

    def _extract_query(self, prompt: str) -> str:
        if "Query: " in prompt:
            lines = prompt.split("Query: ")
            if len(lines) > 1:
                return lines[-1].split("\n")[0].strip()
        lines = [line.strip() for line in prompt.split("\n") if line.strip() and not line.strip().startswith("Query:") and not line.strip().startswith("Category:") and not line.strip().startswith("Return")]
        return lines[-1] if lines else prompt[:100]

    def _classify(self, query: str) -> str:
        q = query.lower()
        intent_keywords = {
            "SIMILARITY": ["similar", "historical parallels", "analogous", "comparable", "resemble", "past events like", "previous crisis like", "similar events", "how is this like"],
            "NEWS": ["news", "latest", "update", "headline", "breaking", "what happened", "sanctions"],
            "SIMULATION": ["simulate", "what if", "scenario", "if happens", "if occurs"],
            "REPORT": ["report", "brief", "intelligence report", "deep dive"],
            "RECOMMENDATION": ["buy", "sell", "invest", "should i", "recommend", "portfolio"],
            "GRAPH": ["relationship", "connection", "how is", "linked to", "connection between", "related to"],
            "MARKET": ["price", "price today", "stock price", "market", "index", "trading", "up today", "down today"],
            "IMPACT": ["impact", "affect", "consequence", "effect", "how does", "why is"],
        }
        for intent, kws in intent_keywords.items():
            if any(kw in q for kw in kws):
                return intent
        if any(k in q for k in ["oil", "energy", "gas", "crude"]):
            return "IMPACT"
        if any(k in q for k in ["stock", "buy", "invest"]):
            return "RECOMMENDATION"
        return "IMPACT"

    def _extract_entities_json(self, query: str) -> str:
        q = query.lower()
        entities = []
        for kw, name in KNOWN_ENTITIES_MAP.items():
            if kw in q and name not in entities:
                entities.append(name)
        for word in q.split():
            w = word.strip(".,;:!?()")
            if w and w[0].isupper() and len(w) > 2 and w.lower() not in {"the","this","what","how","why","when","where","which","that","are","have","been","will","can","all","its","not","but","for","was"}:
                if w not in entities:
                    entities.append(w)
        if not entities:
            if "iran" in q or "israel" in q or "middle east" in q:
                entities = ["Iran", "Israel", "Middle East"]
            elif "russia" in q or "ukraine" in q:
                entities = ["Russia", "Ukraine"]
            elif "china" in q or "taiwan" in q:
                entities = ["China", "Taiwan"]
            elif "oil" in q or "energy" in q:
                entities = ["OPEC", "Saudi Arabia", "Russia"]
            else:
                entities = ["United States", "China", "European Union"]
        return json.dumps(list(set(entities[:6])))

    def _extract_sectors_json(self, query: str) -> str:
        q = query.lower()
        sectors = []
        for sector, kws in SECTOR_KEYWORDS.items():
            if any(kw in q for kw in kws):
                sectors.append(sector.title())
        if not sectors:
            for sector in ["Energy", "Defense"]:
                sectors.append(sector)
        return json.dumps(list(set(sectors[:5])))

    def _extract_tickers_json(self, query: str) -> str:
        tickers = []
        for word in query.split():
            w = word.strip(".,;:!?").upper()
            if len(w) <= 5 and w.isalpha():
                tickers.append(w)
        if not tickers:
            q = query.lower()
            if "energy" in q or "oil" in q:
                tickers = ["XLE", "CVX", "XOM", "BP", "SHEL"]
            elif "defense" in q or "military" in q:
                tickers = ["ITA", "LMT", "NOC", "GD", "RTX"]
            elif "tech" in q or "semiconductor" in q:
                tickers = ["XLK", "QQQ", "NVDA", "AAPL", "MSFT"]
            elif "gold" in q or "safe" in q:
                tickers = ["GLD", "SLV", "GDX"]
            else:
                tickers = ["SPY", "QQQ", "EEM", "XLE"]
        return json.dumps(tickers[:5])

    def _generate_forecast_json(self, query: str) -> str:
        return json.dumps({
            "scenarios": [
                {"name": "Baseline", "probability": 0.55, "impact": "Moderate market impact across affected sectors"},
                {"name": "Escalation", "probability": 0.25, "impact": "Significant disruption, safe-haven rally"},
                {"name": "De-escalation", "probability": 0.20, "impact": "Risk-on rally, recovery in risk assets"},
            ],
            "confidence": 0.72,
            "timeframe": "3-6 months",
        })

    def _generate_report_json(self, query: str) -> str:
        q = query.lower()
        sectors = json.loads(self._extract_sectors_json(query))
        entities = json.loads(self._extract_entities_json(query))
        tickers = json.loads(self._extract_tickers_json(query))
        base_risk = 0.55
        if "oil" in q or "energy" in q or "conflict" in q or "war" in q:
            base_risk = 0.72
        if "sanction" in q or "blockade" in q:
            base_risk = 0.78
        return json.dumps({
            "title": f"Geopolitical Intelligence Report: {query[:60]}",
            "event": query,
            "affected_sectors": sectors,
            "risk_score": base_risk,
            "expected_market_impact": "Moderate to significant impact expected across affected sectors" if base_risk > 0.6 else "Limited market impact anticipated",
            "recommended_assets": tickers[:3],
            "confidence": round(base_risk + 0.08, 2),
            "reasoning": f"Analysis of {', '.join(entities[:3])} indicates geopolitical risk level of {base_risk:.0%}. Key sectors affected: {', '.join(sectors)}.",
            "sources": ["MarketAtlas Intelligence", "Reuters", "Bloomberg"],
        })

    def _handle_json_prompt(self, query: str, full_prompt: str) -> str:
        if "scenario" in full_prompt.lower() and "consequences" in full_prompt.lower():
            return json.dumps({
                "scenario": f"Scenario: {query}",
                "consequences": {"Oil": "+12%", "European Manufacturing": "-5%", "Inflation": "+2.3%"},
                "probability": 0.71,
                "time_horizon": "3-6 months",
                "key_risks": ["Supply chain disruption", "Inflation spike", "Market volatility"],
            })
        if "title" in full_prompt and "risk_score" in full_prompt:
            return self._generate_report_json(query)
        if "scenarios" in full_prompt and "probability" in full_prompt:
            return self._generate_forecast_json(query)
        return self._mock_response(query)

    def _mock_response(self, query: str) -> str:
        q = query.lower()

        if any(kw in q for kw in ["similar", "historical parallels", "analogous", "resemble", "past events like", "what past event"]):
            return self._response_similarity(query)

        if any(kw in q for kw in ["relationship", "connection", "how is", "linked to", "connection between", "related to", "graph"]):
            return self._response_graph(query)

        if any(kw in q for kw in ["buy", "sell", "should i", "recommend", "invest", "portfolio"]):
            return self._response_recommendation(query)

        if any(kw in q for kw in ["simulate", "what if", "scenario", "if happens", "if occurs"]):
            return json.dumps({
                "scenario": f"Scenario: {query}",
                "consequences": {"Oil": "+12%", "European Manufacturing": "-5%", "Inflation": "+2.3%"},
                "probability": 0.71,
                "time_horizon": "3-6 months",
                "key_risks": ["Supply chain disruption", "Inflation spike", "Market volatility"],
            })

        if any(kw in q for kw in ["report", "brief", "intelligence report", "deep dive", "analysis"]):
            return self._generate_report_json(query)

        if any(kw in q for kw in ["oil", "energy", "crude", "gas", "petroleum", "brent", "wti"]):
            return self._response_energy(query)

        if any(kw in q for kw in ["defense", "military", "lockheed", "northrop", "weapon", "ita"]):
            return self._response_defense(query)

        if any(kw in q for kw in ["sanction", "tariff", "trade war", "embargo"]):
            return self._response_sanctions(query)

        if any(kw in q for kw in ["conflict", "war", "attack", "tension", "strike", "blockade", "escalation"]):
            return self._response_conflict(query)

        if any(kw in q for kw in ["market", "stock", "price", "etf", "index", "rally", "decline", "volatile"]):
            return self._response_market(query)

        if any(kw in q for kw in ["tech", "technology", "semiconductor", "chip", "ai"]):
            return self._response_tech(query)

        if any(kw in q for kw in ["gold", "safe haven", "gld"]):
            return self._response_safe_haven(query)

        return self._response_generic(query)

    def _response_similarity(self, query: str) -> str:
        q = query.lower()
        if "iran" in q or "israel" in q or "middle east" in q:
            return (
                "## Historical Event Similarity Analysis\n\n"
                "### Similar Historical Events\n\n"
                "**1. Iran Nuclear Deal / JCPOA (2015)**\n"
                "   - Overall Similarity: **82%**\n"
                "   - Text: 75% | Entity: 88% | Sector: 85%\n"
                "   - Type: Political | Date: 2015-07-14\n"
                "   - Oil prices dropped 10% on deal news, defense stocks fell.\n\n"
                "**2. Strait of Hormuz Crisis (2019)**\n"
                "   - Overall Similarity: **79%**\n"
                "   - Text: 72% | Entity: 85% | Sector: 80%\n"
                "   - Type: Conflict | Date: 2019-06-13\n"
                "   - Oil spiked 15%, shipping insurance costs tripled.\n\n"
                "**3. Iran-Israel Shadow War & Nuclear Escalation (2021-2023)**\n"
                "   - Overall Similarity: **76%**\n"
                "   - Text: 70% | Entity: 82% | Sector: 75%\n"
                "   - Type: Conflict | Date: 2021-04-11\n"
                "   - Energy +12%, Defense +8%, Tech -3%.\n\n"
                "### Aggregated Historical Outcomes\n\n"
                "| Sector | Impact |\n"
                "|--------|--------|\n"
                "| Energy | +12.3% |\n"
                "| Defense | +7.0% |\n"
                "| Technology | -3.0% |\n\n"
                "**Confidence:** 85%"
            )
        if "russia" in q or "ukraine" in q:
            return (
                "## Historical Event Similarity Analysis\n\n"
                "### Similar Historical Events\n\n"
                "**1. Russia-Ukraine War (2022)**\n"
                "   - Overall Similarity: **91%**\n"
                "   - Text: 88% | Entity: 94% | Sector: 90%\n"
                "   - Type: Conflict | Date: 2022-02-24\n"
                "   - Energy +25%, Defense +15%, Agriculture +10%.\n\n"
                "**2. Crimea Annexation (2014)**\n"
                "   - Overall Similarity: **74%**\n"
                "   - Text: 68% | Entity: 80% | Sector: 75%\n"
                "   - Type: Conflict | Date: 2014-03-18\n"
                "   - Energy +8%, Defense +5%, Financials -4%.\n\n"
                "**3. European Energy Crisis (2021-2022)**\n"
                "   - Overall Similarity: **67%**\n"
                "   - Text: 62% | Entity: 70% | Sector: 70%\n"
                "   - Type: Economic | Date: 2021-10-01\n"
                "   - Gas prices +400%, Manufacturing -8%.\n\n"
                "**Confidence:** 87%"
            )
        return (
            "## Historical Event Similarity Analysis\n\n"
            "### Similar Historical Events\n\n"
            "**1. Russia-Ukraine War (2022)**\n"
            "   - Overall Similarity: **67%**\n"
            "   - Text: 63% | Entity: 71% | Sector: 65%\n"
            "   - Energy +25%, Defense +15%.\n\n"
            "**2. Global Financial Crisis (2008)**\n"
            "   - Overall Similarity: **52%**\n"
            "   - Text: 48% | Entity: 55% | Sector: 55%\n"
            "   - Financials -40%, Real Estate -50%.\n\n"
            "**3. COVID-19 Pandemic (2020)**\n"
            "   - Overall Similarity: **45%**\n"
            "   - Text: 42% | Entity: 48% | Sector: 45%\n"
            "   - Healthcare +15%, Travel -30%.\n\n"
            "**Confidence:** 72%"
        )

    def _response_graph(self, query: str) -> str:
        q = query.lower()
        entities = json.loads(self._extract_entities_json(query))
        ent_text = ", ".join(entities[:4])
        if "russia" in q and "europe" in q:
            return (
                f"## Knowledge Graph Analysis\n\n"
                f"### Entities: {ent_text}\n\n"
                f"**Connection Path:**\n\n"
                f"Russia\n"
                f"  ├── supplies → Natural Gas (40% of EU imports)\n"
                f"  ├── supplies → Oil (25% of EU imports)\n"
                f"  ├── pipeline → Nord Stream (Germany)\n"
                f"  ├── pipeline → TurkStream (Southern Europe)\n"
                f"  └── rivalry → NATO (baltic states, eastern flank)\n\n"
                f"European Union\n"
                f"  ├── imports → Russian Energy ($400B/year)\n"
                f"  ├── sanctions → Russia (2022-present)\n"
                f"  ├── diversification → LNG from US/Qatar\n"
                f"  └── impact → Manufacturing (-5%), Inflation (+3%)\n\n"
                f"**Key Insight:** Russia supplies ~40% of EU natural gas and ~25% of EU oil. "
                f"The Russia-Ukraine conflict has disrupted these flows, driving European energy prices up 200%+ "
                f"and forcing a rapid diversification to LNG imports and renewable energy."
            )
        return (
            f"## Knowledge Graph Analysis\n\n"
            f"### Entities: {ent_text}\n\n"
            f"**Relationships Identified:**\n\n"
            f"{entities[0] if len(entities) > 0 else 'Entity'} ↔ "
            f"{entities[1] if len(entities) > 1 else 'Global Markets'}\n"
            f"  ├── trade relationship\n"
            f"  ├── geopolitical tension\n"
            f"  └── market impact channel\n\n"
            f"**Cascading Effects:**\n"
            f"1. Primary: Direct impact on energy and defense sectors\n"
            f"2. Secondary: Supply chain reconfiguration\n"
            f"3. Tertiary: Inflation and interest rate implications\n\n"
            f"**Confidence:** 78% based on available relationship data."
        )

    def _response_recommendation(self, query: str) -> str:
        q = query.lower()
        if "energy" in q or "oil" in q:
            return (
                "## Investment Recommendation\n\n"
                "**Action:** BUY | **Sector:** Energy | **Conviction:** High\n\n"
                "**Rationale:**\n"
                "1. Geopolitical risk premium supporting elevated oil prices\n"
                "2. Underinvestment in new supply creating structural deficit\n"
                "3. Energy sector valuations attractive relative to broader market\n\n"
                "**Top Picks:** XLE (Energy Select Sector SPDR), CVX, XOM\n\n"
                "**Key Risks:** Global recession, demand destruction, OPEC+ discord\n\n"
                "**Confidence:** 82%"
            )
        if "defense" in q or "military" in q:
            return (
                "## Investment Recommendation\n\n"
                "**Action:** BUY | **Sector:** Defense | **Conviction:** High\n\n"
                "**Rationale:**\n"
                "1. Rising global defense budgets (NATO 2% target, Asia buildup)\n"
                "2. Long-term secular growth regardless of economic cycle\n"
                "3. Strong order books and multi-year backlogs\n\n"
                "**Top Picks:** ITA (Defense ETF), LMT, NOC\n\n"
                "**Key Risks:** Budget cuts, contract delays, geopolitical de-escalation\n\n"
                "**Confidence:** 80%"
            )
        if "tech" in q or "semiconductor" in q or "ai" in q:
            return (
                "## Investment Recommendation\n\n"
                "**Action:** HOLD | **Sector:** Technology | **Conviction:** Medium\n\n"
                "**Rationale:**\n"
                "1. AI investment cycle providing strong tailwinds\n"
                "2. Valuation multiples elevated, limiting upside\n"
                "3. Regulatory risks increasing globally\n\n"
                "**Top Picks:** XLK (Technology Select Sector SPDR), NVDA, MSFT\n\n"
                "**Key Risks:** Regulatory crackdown, valuation compression, supply chain disruption\n\n"
                "**Confidence:** 72%"
            )
        if "safe" in q or "gold" in q or "haven" in q:
            return (
                "## Investment Recommendation\n\n"
                "**Action:** BUY | **Sector:** Precious Metals | **Conviction:** Medium\n\n"
                "**Rationale:**\n"
                "1. Geopolitical uncertainty driving safe-haven demand\n"
                "2. Central bank gold purchasing at record levels\n"
                "3. Real yields supportive for gold prices\n\n"
                "**Top Picks:** GLD (Gold ETF), GDX (Gold Miners)\n\n"
                "**Key Risks:** Fed hawkishness, dollar strength, risk-on rotation\n\n"
                "**Confidence:** 75%"
            )
        return (
            "## Investment Recommendation\n\n"
            "**Action:** HOLD | **Conviction:** Medium\n\n"
            "**Rationale:**\n"
            "Current market conditions suggest a cautiously defensive posture. "
            "Geopolitical risks remain elevated while valuations are mixed across sectors.\n\n"
            "**Recommended Allocation:**\n"
            "1. 30% Energy (XLE) — geopolitical risk premium\n"
            "2. 25% Defense (ITA) — secular budget growth\n"
            "3. 20% Gold (GLD) — safe-haven allocation\n"
            "4. 15% Cash — optionality for dislocations\n"
            "5. 10% Tech (XLK) — selective AI exposure\n\n"
            "**Confidence:** 68%"
        )

    def _response_energy(self, query: str) -> str:
        q = query.lower()
        iran_mideast = any(k in q for k in ["iran", "israel", "middle east", "hormuz", "gulf"])
        russia = any(k in q for k in ["russia", "ukraine", "nord stream"])
        details = ""
        if iran_mideast:
            details = " Primary driver: Middle East tensions threatening Strait of Hormuz transit (20% of global supply). Market is pricing in a 15-20% disruption risk premium."
        elif russia:
            details = " Primary driver: Russia-Ukraine conflict disrupting Russian exports and European energy security. Sanctions have redirected global energy trade flows."
        return (
            f"## Energy Market Analysis\n\n"
            f"**Assessment:** Energy markets experiencing significant upward pressure due to geopolitical tensions.\n\n"
            f"**Key Drivers:**\n"
            f"- Supply constraints from sanctioned/restricted production\n"
            f"- Rising shipping and insurance costs\n"
            f"- Elevated geopolitical risk premium in futures pricing\n"
            f"- Strategic reserve drawdowns reducing buffer capacity\n\n"
            f"**Sector Impact:**\n"
            f"- Upstream producers: Positive (higher prices, strong margins)\n"
            f"- Refiners: Mixed (higher input costs, product demand resilient)\n"
            f"- Transportation: Negative (higher fuel costs impacting margins)\n\n"
            f"**Price Outlook:** Elevated volatility expected with upside bias.\n"
            f" Key levels: Brent $75-85 near-term, $85-100 if supply disruption materializes."
            f"{details}\n\n"
            f"**Confidence:** 85%"
        )

    def _response_defense(self, query: str) -> str:
        return (
            "## Defense Sector Analysis\n\n"
            "**Assessment:** Defense sector outlook strengthened by elevated geopolitical tensions globally.\n\n"
            "**Key Drivers:**\n"
            "1. NATO members increasing defense spending toward 2%+ GDP target\n"
            "2. Asia Pacific military buildup (Japan, Korea, Australia, Taiwan)\n"
            "3. Middle East modernization programs (Saudi Arabia, UAE, Israel)\n"
            "4. Long-term secular growth independent of economic cycles\n\n"
            "**Sub-sector Analysis:**\n"
            "- Aerospace & Missiles: Strong demand, multi-year order books\n"
            "- Cybersecurity: Rapid growth from government/commercial demand\n"
            "- Naval Shipbuilding: Modernization programs driving sustained spending\n"
            "- C4ISR: Intelligence and surveillance spending accelerating\n\n"
            "**Confidence:** 80%"
        )

    def _response_sanctions(self, query: str) -> str:
        return (
            "## Sanctions Impact Analysis\n\n"
            "**Assessment:** Sanctions are creating significant market dislocations and supply chain reconfiguration.\n\n"
            "**Affected Sectors:**\n"
            "- Energy: Russian oil/gas exports redirected, price cap mechanisms creating friction\n"
            "- Financials: SWIFT disconnection, correspondent banking restrictions\n"
            "- Shipping: Insurance costs up 300%+ for sanctioned route exposure\n"
            "- Manufacturing: Critical material shortages (titanium, neon, nickel)\n\n"
            "**Market Implications:**\n"
            "1. Inflationary pressure from supply constraints\n"
            "2. Trade flow reconfiguration (Asia-Middle East corridors strengthening)\n"
            "3. De-dollarization acceleration in trade settlement\n"
            "4. Second-order effects on emerging market debt\n\n"
            "**Confidence:** 82%"
        )

    def _response_conflict(self, query: str) -> str:
        q = query.lower()
        region = "Middle East" if any(k in q for k in ["iran", "israel", "middle east", "hormuz", "red sea", "gulf"]) else \
                 "Eastern Europe" if any(k in q for k in ["russia", "ukraine", "crimea", "donbas"]) else \
                 "Asia Pacific" if any(k in q for k in ["taiwan", "china", "south china", "korea"]) else \
                 "Global"
        return (
            f"## Geopolitical Risk Assessment\n\n"
            f"**Region:** {region}\n"
            f"**Risk Level:** HIGH\n\n"
            f"**Assessment:** Elevated geopolitical tensions detected with potential for significant market disruption.\n\n"
            f"**Direct Market Impacts Expected:**\n"
            f"1. Energy prices: Upward pressure (risk premium + supply concerns)\n"
            f"2. Defense stocks: Positive catalyst (budget increase expectations)\n"
            f"3. Safe-haven assets: Increased demand (gold, USD, government bonds)\n"
            f"4. Risk assets: Near-term selling pressure\n\n"
            f"**Scenario Analysis:**\n"
            f"- Base case (55%): Elevated tensions, limited escalation → moderate market impact\n"
            f"- Escalation case (25%): Active conflict → significant disruption\n"
            f"- De-escalation case (20%): Diplomatic resolution → risk-on rally\n\n"
            f"**Key Indicators to Monitor:**\n"
            f"- Diplomatic channels and statements\n"
            f"- Military posture changes\n"
            f"- Energy and shipping insurance markets\n"
            f"- Safe-haven asset flows\n\n"
            f"**Confidence:** 78% based on current intelligence indicators."
        )

    def _response_market(self, query: str) -> str:
        return (
            "## Market Analysis\n\n"
            "**Assessment:** Markets showing mixed signals with elevated geopolitical risk weighing on sentiment.\n\n"
            "**Current Conditions:**\n"
            "- Momentum: Mixed — defensive sectors outperforming cyclicals\n"
            "- Volatility: Elevated (VIX above historical average)\n"
            "- Volume: Above average with institutional positioning\n"
            "- Breadth: Narrow leadership, primarily in energy and defense\n\n"
            "**Sector Rotation:**\n"
            "→ Inflows: Energy (+), Defense (+), Gold (+), Utilities (+)\n"
            "→ Outflows: Tech (-), Consumer Discretionary (-), Real Estate (-)\n\n"
            "**Technical Levels:**\n"
            "- S&P 500: Support at 4,200, resistance at 4,500\n"
            "- 10Y Yield: Range-bound 3.8-4.2%\n"
            "- VIX: Elevated above 20 suggesting continued uncertainty\n\n"
            "**Confidence:** 75%"
        )

    def _response_tech(self, query: str) -> str:
        return (
            "## Technology Sector Analysis\n\n"
            "**Assessment:** Technology sector facing headwinds from geopolitical tensions and regulatory scrutiny.\n\n"
            "**Key Themes:**\n"
            "1. AI investment cycle driving capex spending by hyperscalers\n"
            "2. Semiconductor supply chain diversification (Chip 4, CHIPS Act)\n"
            "3. China tech regulatory environment remains challenging\n"
            "4. Cybersecurity spending accelerating across all verticals\n\n"
            "**Sub-sector Outlook:**\n"
            "- AI/ML: Strong growth, high valuation multiples\n"
            "- Semiconductors: Cyclical recovery underway, AI-driven demand\n"
            "- Cloud: Secular growth but decelerating from pandemic peaks\n"
            "- Cybersecurity: Defensive growth with geopolitical tailwinds\n\n"
            "**Key Risk:** Taiwan Strait tensions threatening semiconductor supply chain (60% of advanced chips from TSMC).\n\n"
            "**Confidence:** 72%"
        )

    def _response_safe_haven(self, query: str) -> str:
        return (
            "## Safe-Haven Asset Analysis\n\n"
            "**Assessment:** Safe-haven demand increasing amid geopolitical uncertainty.\n\n"
            "**Gold:**\n"
            "- Central bank purchases at 50-year highs (1,000+ tonnes/year)\n"
            "- De-dollarization trend supporting structural demand\n"
            "- Key level: $2,000-2,200 near-term, upside bias\n\n"
            "**Government Bonds:**\n"
            "- Flight-to-safety flows supporting treasury prices\n"
            "- Yield curve signaling growth concerns\n"
            "- Real yields attractive at current levels\n\n"
            "**Other Safe Havens:**\n"
            "- USD: Strength from risk-off flows and rate differential\n"
            "- CHF, JPY: Traditional safe-haven currencies appreciating\n"
            "- Gold miners: Operating leverage to higher gold prices\n\n"
            "**Confidence:** 80%"
        )

    def _response_generic(self, query: str) -> str:
        entities = json.loads(self._extract_entities_json(query))
        sectors = json.loads(self._extract_sectors_json(query))
        return (
            f"## Geopolitical Intelligence Analysis\n\n"
            f"**Query:** {query}\n\n"
            f"**Entities Identified:** {', '.join(entities)}\n"
            f"**Affected Sectors:** {', '.join(sectors)}\n\n"
            f"**Assessment:**\n"
            f"Analysis of the current geopolitical landscape involving "
            f"{', '.join(entities[:3])} indicates moderate to elevated risk levels "
            f"with potential implications across {', '.join(sectors[:3])} sectors.\n\n"
            f"**Key Factors:**\n"
            f"1. Geopolitical tensions creating uncertainty in affected regions\n"
            f"2. Supply chain implications for global trade flows\n"
            f"3. Market pricing in risk premium across relevant asset classes\n"
            f"4. Policy responses likely to influence near-term outcomes\n\n"
            f"**Recommendation:** Monitor diplomatic developments and sector-specific indicators. "
            f"Consider defensive positioning with exposure to energy, defense, and safe-haven assets "
            f"if geopolitical risks continue to escalate.\n\n"
            f"**Confidence:** 72% based on available intelligence."
        )

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
        except Exception:
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
