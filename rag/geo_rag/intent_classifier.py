from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GeoIntent(str, Enum):
    NEWS_ANALYSIS = "news_analysis"
    HISTORICAL_ANALOG = "historical_analog"
    MARKET_IMPACT = "market_impact"
    GRAPH_RELATIONSHIP = "graph_relationship"
    GEOPOLITICAL_RISK = "geopolitical_risk"
    GENERAL_QUERY = "general_query"
    MULTI_SOURCE = "multi_source"


@dataclass
class IntentResult:
    intent: GeoIntent
    confidence: float
    sub_intent: str = ""
    extracted_entities: List[str] = field(default_factory=list)
    extracted_sectors: List[str] = field(default_factory=list)
    extracted_regions: List[str] = field(default_factory=list)


INTENT_PATTERNS: Dict[GeoIntent, List[str]] = {
    GeoIntent.NEWS_ANALYSIS: [
        "news", "latest", "update", "headline", "what happened", "breaking",
        "report on", "coverage", "announcement", "developments", "situation",
    ],
    GeoIntent.HISTORICAL_ANALOG: [
        "similar to", "historical parallels", "resemble", "analogous", "comparable",
        "past event", "like what happened", "previous crisis", "history",
        "historical example", "analogy", "reminiscent of", "echoes of",
        "what historical", "similar event", "comparable situation",
    ],
    GeoIntent.MARKET_IMPACT: [
        "market", "stock", "price", "etf", "index", "trading", "invest",
        "portfolio", "buy", "sell", "hold", "sector", "oil price", "gold",
        "nifty", "sensex", "s&p", "nasdaq", "commodity", "futures",
        "how will", "affect", "impact on", "reaction of", "effect on",
    ],
    GeoIntent.GRAPH_RELATIONSHIP: [
        "relationship", "connection", "how is", "linked to", "connection between",
        "relates to", "dependence", "dependency", "supply chain", "depends on",
        "path from", "network", "how does", "affect", "ties between",
    ],
    GeoIntent.GEOPOLITICAL_RISK: [
        "risk", "threat", "danger", "escalation", "conflict", "war", "tension",
        "crisis", "instability", "sanctions", "military", "invasion",
        "geopolitical", "strategy", "security", "defense",
    ],
    GeoIntent.MULTI_SOURCE: [
        "comprehensive", "overview", "analysis", "tell me about", "explain",
        "what is", "what are", "how", "why", "deep dive", "full picture",
        "complete", "thorough", "detailed", "synthesis",
    ],
}

SECTOR_KEYWORDS = {
    "energy": ["oil", "gas", "energy", "petroleum", "brent", "crude", "lng", "coal", "renewable", "opec"],
    "technology": ["tech", "semiconductor", "chip", "nvidia", "tsmc", "software", "ai", "data", "cyber"],
    "defense": ["defense", "military", "weapon", "nato", "army", "navy", "air force", "missile"],
    "finance": ["bank", "finance", "stock", "market", "etf", "bond", "currency", "exchange", "rate"],
    "shipping": ["shipping", "port", "canal", "strait", "trade route", "freight", "container"],
    "agriculture": ["agriculture", "food", "wheat", "grain", "corn", "soybean", "fertilizer"],
}

REGION_KEYWORDS = {
    "Middle East": ["iran", "iraq", "saudi", "israel", "gaza", "hormuz", "middle east", "gulf", "opec"],
    "Europe": ["europe", "eu", "nato", "ukraine", "russia", "germany", "france", "uk", "brexit"],
    "Asia Pacific": ["china", "taiwan", "japan", "korea", "south china sea", "india", "pacific", "asean"],
    "Global": ["global", "world", "international", "worldwide"],
    "North America": ["usa", "united states", "america", "canada", "mexico"],
    "Africa": ["africa", "libya", "nigeria", "suez", "egypt"],
    "South America": ["venezuela", "brazil", "argentina", "chile"],
}

ENTITY_KEYWORDS = {
    "Iran": ["iran", "tehran", "ayatollah", "irgc"],
    "Russia": ["russia", "moscow", "putin", "kremlin"],
    "China": ["china", "beijing", "xi jinping", "ccp"],
    "USA": ["usa", "united states", "america", "washington", "white house", "us"],
    "Taiwan": ["taiwan", "taipei", "tsmc"],
    "Ukraine": ["ukraine", "kyiv", "zelensky"],
    "Israel": ["israel", "tel aviv", "netanyahu"],
    "Saudi Arabia": ["saudi", "riyadh", "mbs"],
    "Europe": ["europe", "eu", "brussels", "european union"],
    "NATO": ["nato"],
    "OPEC": ["opec"],
    "Hamas": ["hamas", "gaza"],
}


class GeoIntentClassifier:
    def __init__(self):
        self._cache = {}

    def get_intent_labels(self) -> dict:
        return {intent.value: intent.name for intent in GeoIntent}

    def get_supported_sectors(self) -> list:
        return list(SECTOR_KEYWORDS.keys())

    def get_supported_regions(self) -> list:
        return list(REGION_KEYWORDS.keys())

    def classify(self, query: str) -> IntentResult:
        query_lower = query.lower().strip()
        intent_scores: Dict[GeoIntent, float] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if pattern in query_lower:
                    score += 1.0
            if score > 0:
                intent_scores[intent] = score
        if not intent_scores:
            if len(query_lower.split()) >= 4:
                intent_scores[GeoIntent.MULTI_SOURCE] = 0.5
            else:
                intent_scores[GeoIntent.GENERAL_QUERY] = 0.5
        best_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[best_intent]
        total_score = sum(intent_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        if len(intent_scores) >= 3:
            best_intent = GeoIntent.MULTI_SOURCE
            confidence = min(confidence + 0.2, 1.0)
        extracted_entities = []
        for entity, keywords in ENTITY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                extracted_entities.append(entity)
        extracted_sectors = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                extracted_sectors.append(sector)
        extracted_regions = []
        for region, keywords in REGION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                extracted_regions.append(region)
        return IntentResult(
            intent=best_intent,
            confidence=min(confidence, 1.0),
            sub_intent=self._determine_sub_intent(best_intent, query_lower),
            extracted_entities=extracted_entities,
            extracted_sectors=extracted_sectors,
            extracted_regions=extracted_regions,
        )

    def _determine_sub_intent(self, intent: GeoIntent, query_lower: str) -> str:
        if intent == GeoIntent.NEWS_ANALYSIS:
            if "sanction" in query_lower:
                return "sanctions"
            if "conflict" in query_lower or "war" in query_lower:
                return "conflict"
            if "trade" in query_lower:
                return "trade"
            return "general_news"
        if intent == GeoIntent.HISTORICAL_ANALOG:
            return "find_analogs"
        if intent == GeoIntent.MARKET_IMPACT:
            if "oil" in query_lower:
                return "energy_market"
            if "stock" in query_lower or "etf" in query_lower:
                return "equity_market"
            if "gold" in query_lower:
                return "safe_haven"
            return "general_market"
        if intent == GeoIntent.GRAPH_RELATIONSHIP:
            return "graph_query"
        if intent == GeoIntent.GEOPOLITICAL_RISK:
            return "risk_assessment"
        return "general"
