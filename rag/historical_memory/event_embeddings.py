from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from rag.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


HISTORICAL_EVENTS = [
    {
        "name": "Iran Nuclear Deal Collapse (2018)",
        "description": "US withdrew from JCPOA, reimposed sanctions on Iran, disrupting global oil markets and Middle Eastern geopolitics.",
        "date": "2018-05-08",
        "entities": ["Iran", "USA", "Europe", "OPEC"],
        "sectors": ["energy", "finance"],
        "regions": ["Middle East", "Europe"],
        "impact": "high",
        "market_effect": "oil prices surged 25% in 3 months",
    },
    {
        "name": "Russia-Ukraine War (2022)",
        "description": "Full-scale Russian invasion caused energy crisis, commodity spikes, and global realignment of supply chains.",
        "date": "2022-02-24",
        "entities": ["Russia", "Ukraine", "NATO", "EU", "USA", "China"],
        "sectors": ["energy", "defense", "agriculture", "finance"],
        "regions": ["Europe", "Global"],
        "impact": "severe",
        "market_effect": "gas prices +400%, oil +60%",
    },
    {
        "name": "Taiwan Straits Crisis (2022)",
        "description": "Chinese military exercises around Taiwan following Pelosi visit, threatening semiconductor supply chains.",
        "date": "2022-08-02",
        "entities": ["China", "Taiwan", "USA", "Japan"],
        "sectors": ["technology", "semiconductor", "defense"],
        "regions": ["Asia Pacific", "Global"],
        "impact": "moderate",
        "market_effect": "SOX -15% then recovered in 2 weeks",
    },
    {
        "name": "Hormuz Strait Disruption (2019)",
        "description": "Tanker attacks and US-Iran tensions threatened the world's most important oil chokepoint.",
        "date": "2019-06-13",
        "entities": ["Iran", "USA", "Saudi Arabia", "UAE", "Japan"],
        "sectors": ["energy", "shipping"],
        "regions": ["Middle East", "Global"],
        "impact": "high",
        "market_effect": "oil +15% in 1 week",
    },
    {
        "name": "US-China Trade War (2018-2019)",
        "description": "Tariff escalation between world's two largest economies disrupted global trade and supply chains.",
        "date": "2018-07-06",
        "entities": ["USA", "China", "EU"],
        "sectors": ["technology", "manufacturing", "agriculture"],
        "regions": ["Global"],
        "impact": "moderate",
        "market_effect": "S&P -10% over 3 months",
    },
    {
        "name": "Gulf War (1990-1991)",
        "description": "Iraqi invasion of Kuwait led to oil price spike and US-led military intervention.",
        "date": "1990-08-02",
        "entities": ["Iraq", "Kuwait", "USA", "Saudi Arabia", "UN"],
        "sectors": ["energy", "defense"],
        "regions": ["Middle East"],
        "impact": "severe",
        "market_effect": "oil prices doubled from $15 to $30",
    },
    {
        "name": "COVID-19 Pandemic (2020)",
        "description": "Global pandemic caused unprecedented economic shutdown, market crash, and stimulus response.",
        "date": "2020-03-11",
        "entities": ["WHO", "USA", "China", "EU"],
        "sectors": ["healthcare", "finance", "technology"],
        "regions": ["Global"],
        "impact": "severe",
        "market_effect": "S&P -34% in 1 month",
    },
    {
        "name": "Hamas Attack on Israel (2023)",
        "description": "Major escalation in Middle East conflict threatening regional stability and energy routes.",
        "date": "2023-10-07",
        "entities": ["Israel", "Hamas", "Iran", "USA", "Egypt"],
        "sectors": ["energy", "defense"],
        "regions": ["Middle East"],
        "impact": "high",
        "market_effect": "oil +6% in first week",
    },
    {
        "name": "OPEC Oil Crisis (1973)",
        "description": "Arab oil embargo against US and allies caused quadrupling of oil prices and economic turmoil.",
        "date": "1973-10-17",
        "entities": ["OPEC", "USA", "Europe", "Japan", "Israel"],
        "sectors": ["energy", "finance"],
        "regions": ["Middle East", "Global"],
        "impact": "severe",
        "market_effect": "oil prices quadrupled",
    },
    {
        "name": "Annexation of Crimea (2014)",
        "description": "Russian annexation led to Western sanctions and first signs of European energy weaponization.",
        "date": "2014-03-18",
        "entities": ["Russia", "Ukraine", "Crimea", "EU", "USA", "NATO"],
        "sectors": ["energy", "defense", "finance"],
        "regions": ["Europe"],
        "impact": "moderate",
        "market_effect": "Russian market -15%, gas fears",
    },
]


class EventEmbedder:
    def __init__(self):
        self.embedder = get_embedding_model()
        self._embeddings: Dict[str, np.ndarray] = {}
        self._build_embeddings()

    def _build_embeddings(self):
        for event in HISTORICAL_EVENTS:
            text = self._event_to_text(event)
            vec = self.embedder.embed([text])[0]
            self._embeddings[event["name"]] = vec

    def _event_to_text(self, event: dict) -> str:
        return f"{event['name']}: {event['description']} Sectors: {', '.join(event['sectors'])}. Regions: {', '.join(event['regions'])}. Entities: {', '.join(event['entities'])}."

    def get_embedding(self, event_name: str) -> Optional[np.ndarray]:
        return self._embeddings.get(event_name)

    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        return self._embeddings

    def embed_text(self, text: str) -> np.ndarray:
        return self.embedder.embed([text])[0]
