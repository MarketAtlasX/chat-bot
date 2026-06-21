from __future__ import annotations

import logging
from typing import List, Optional

from rag.embeddings import get_embedding_model
from rag.retrievers.base import BaseRetriever, RetrievalResult, RetrieverType
from rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)

HISTORICAL_EVENTS = [
    {
        "name": "Iran Nuclear Deal Collapse (2018)",
        "description": "US withdrew from JCPOA, reimposed sanctions on Iran, disrupting global oil markets and Middle Eastern geopolitics.",
        "date": "2018-05-08",
        "entities": ["Iran", "USA", "Europe", "OPEC"],
        "sectors": ["energy", "finance"],
        "regions": ["Middle East", "Europe"],
    },
    {
        "name": "Russia-Ukraine War (2022)",
        "description": "Full-scale Russian invasion caused energy crisis, commodity spikes, and global realignment of supply chains.",
        "date": "2022-02-24",
        "entities": ["Russia", "Ukraine", "NATO", "EU", "USA", "China"],
        "sectors": ["energy", "defense", "agriculture", "finance"],
        "regions": ["Europe", "Global"],
    },
    {
        "name": "Taiwan Straits Crisis (2022)",
        "description": "Chinese military exercises around Taiwan following Pelosi visit, threatening semiconductor supply chains.",
        "date": "2022-08-02",
        "entities": ["China", "Taiwan", "USA", "Japan"],
        "sectors": ["technology", "semiconductor", "defense"],
        "regions": ["Asia Pacific", "Global"],
    },
    {
        "name": "Hormuz Strait Disruption (2019)",
        "description": "Tanker attacks and US-Iran tensions threatened the world's most important oil chokepoint.",
        "date": "2019-06-13",
        "entities": ["Iran", "USA", "Saudi Arabia", "UAE", "Japan"],
        "sectors": ["energy", "shipping"],
        "regions": ["Middle East", "Global"],
    },
    {
        "name": "US-China Trade War (2018-2019)",
        "description": "Tariff escalation between world's two largest economies disrupted global trade and supply chains.",
        "date": "2018-07-06",
        "entities": ["USA", "China", "EU"],
        "sectors": ["technology", "manufacturing", "agriculture"],
        "regions": ["Global"],
    },
    {
        "name": "Gulf War (1990-1991)",
        "description": "Iraqi invasion of Kuwait led to oil price spike and US-led military intervention.",
        "date": "1990-08-02",
        "entities": ["Iraq", "Kuwait", "USA", "Saudi Arabia", "UN"],
        "sectors": ["energy", "defense"],
        "regions": ["Middle East"],
    },
    {
        "name": "COVID-19 Pandemic (2020)",
        "description": "Global pandemic caused unprecedented economic shutdown, market crash, and stimulus response.",
        "date": "2020-03-11",
        "entities": ["WHO", "USA", "China", "EU"],
        "sectors": ["healthcare", "finance", "technology"],
        "regions": ["Global"],
    },
    {
        "name": "Suez Canal Blockage (2021)",
        "description": "Ever Given container ship blocked Suez Canal for 6 days, disrupting global shipping.",
        "date": "2021-03-23",
        "entities": ["Egypt", "Global trade"],
        "sectors": ["shipping", "trade"],
        "regions": ["Global"],
    },
    {
        "name": "Libyan Civil War (2011)",
        "description": "Civil war led to collapse of Libyan oil production and NATO military intervention.",
        "date": "2011-02-15",
        "entities": ["Libya", "NATO", "France", "UK"],
        "sectors": ["energy", "defense"],
        "regions": ["Middle East", "Africa"],
    },
    {
        "name": "Korean Peninsula Tensions (2017)",
        "description": "North Korean missile tests and US military posturing raised fears of conflict.",
        "date": "2017-09-03",
        "entities": ["North Korea", "South Korea", "USA", "Japan", "China"],
        "sectors": ["defense", "technology"],
        "regions": ["Asia Pacific"],
    },
    {
        "name": "Annexation of Crimea (2014)",
        "description": "Russian annexation led to Western sanctions and first signs of European energy weaponization.",
        "date": "2014-03-18",
        "entities": ["Russia", "Ukraine", "Crimea", "EU", "USA", "NATO"],
        "sectors": ["energy", "defense", "finance"],
        "regions": ["Europe"],
    },
    {
        "name": "OPEC Oil Crisis (1973)",
        "description": "Arab oil embargo against US and allies caused quadrupling of oil prices and economic turmoil.",
        "date": "1973-10-17",
        "entities": ["OPEC", "USA", "Europe", "Japan", "Israel"],
        "sectors": ["energy", "finance"],
        "regions": ["Middle East", "Global"],
    },
    {
        "name": "Japan Tsunami & Fukushima (2011)",
        "description": "Earthquake and tsunami caused nuclear disaster, disrupting global energy and supply chains.",
        "date": "2011-03-11",
        "entities": ["Japan", "Global nuclear industry"],
        "sectors": ["energy", "technology", "manufacturing"],
        "regions": ["Asia Pacific", "Global"],
    },
    {
        "name": "Fall of the Berlin Wall (1989)",
        "description": "Collapse of Soviet bloc led to market integration of Eastern Europe and new trade dynamics.",
        "date": "1989-11-09",
        "entities": ["USSR", "East Germany", "West Germany", "USA", "EU"],
        "sectors": ["finance", "trade"],
        "regions": ["Europe"],
    },
    {
        "name": "Asian Financial Crisis (1997)",
        "description": "Currency collapses across Southeast Asia triggered global financial contagion.",
        "date": "1997-07-02",
        "entities": ["Thailand", "South Korea", "Indonesia", "IMF", "USA"],
        "sectors": ["finance", "trade"],
        "regions": ["Asia Pacific", "Global"],
    },
    {
        "name": "9/11 Attacks (2001)",
        "description": "Terrorist attacks led to global security realignment, wars in Afghanistan/Iraq, and market disruption.",
        "date": "2001-09-11",
        "entities": ["USA", "Al-Qaeda", "Afghanistan", "NATO"],
        "sectors": ["defense", "finance", "energy", "aviation"],
        "regions": ["Global"],
    },
    {
        "name": "Hamas Attack on Israel (2023)",
        "description": "Major escalation in Middle East conflict threatening regional stability and energy routes.",
        "date": "2023-10-07",
        "entities": ["Israel", "Hamas", "Iran", "USA", "Egypt"],
        "sectors": ["energy", "defense"],
        "regions": ["Middle East"],
    },
    {
        "name": "Brexit Referendum (2016)",
        "description": "UK voted to leave EU, causing political and market uncertainty across Europe.",
        "date": "2016-06-23",
        "entities": ["UK", "EU", "Scotland"],
        "sectors": ["finance", "trade"],
        "regions": ["Europe"],
    },
    {
        "name": "South China Sea Tensions (2016)",
        "description": "Permanent Court of Arbitration ruling against China's claims escalated regional military tensions.",
        "date": "2016-07-12",
        "entities": ["China", "Philippines", "USA", "ASEAN"],
        "sectors": ["defense", "shipping", "trade"],
        "regions": ["Asia Pacific"],
    },
    {
        "name": "Sri Lanka Economic Collapse (2022)",
        "description": "Debt default and political crisis highlighted emerging market vulnerabilities to global shocks.",
        "date": "2022-04-12",
        "entities": ["Sri Lanka", "IMF", "China", "India"],
        "sectors": ["finance", "trade"],
        "regions": ["South Asia"],
    },
    {
        "name": "US Debt Ceiling Crisis (2011)",
        "description": "Political deadlock over debt ceiling led to first US credit rating downgrade and market volatility.",
        "date": "2011-08-05",
        "entities": ["USA", "S&P"],
        "sectors": ["finance"],
        "regions": ["North America", "Global"],
    },
    {
        "name": "Cyprus Financial Crisis (2013)",
        "description": "Banking collapse in Cyprus led to bail-in of depositors and Eurozone contagion fears.",
        "date": "2013-03-16",
        "entities": ["Cyprus", "EU", "ECB", "IMF"],
        "sectors": ["finance"],
        "regions": ["Europe"],
    },
    {
        "name": "Venezuela Collapse (2014-ongoing)",
        "description": "Economic collapse and political crisis in major oil-producing nation affected global heavy crude markets.",
        "date": "2014-01-01",
        "entities": ["Venezuela", "USA", "OPEC"],
        "sectors": ["energy", "finance"],
        "regions": ["South America"],
    },
    {
        "name": "China Evergrande Crisis (2021)",
        "description": "Collapse of China's second-largest developer triggered fears of systemic financial contagion.",
        "date": "2021-09-20",
        "entities": ["China", "Evergrande", "Global markets"],
        "sectors": ["finance", "real estate"],
        "regions": ["Asia Pacific", "Global"],
    },
    {
        "name": "Global Chip Shortage (2021-2023)",
        "description": "Supply chain disruption and surging demand caused global semiconductor shortage affecting multiple industries.",
        "date": "2021-01-01",
        "entities": ["TSMC", "Samsung", "USA", "China", "Taiwan"],
        "sectors": ["technology", "semiconductor", "automotive"],
        "regions": ["Global"],
    },
    {
        "name": "AUKUS Pact (2021)",
        "description": "Trilateral security pact between Australia, UK, and US reshaped Indo-Pacific defense dynamics.",
        "date": "2021-09-15",
        "entities": ["Australia", "UK", "USA", "China", "France"],
        "sectors": ["defense", "technology"],
        "regions": ["Asia Pacific", "Europe"],
    },
]


class HistoricalRetriever(BaseRetriever):
    def __init__(self, collection: str = "marketatlas_historical"):
        super().__init__(name="historical_retriever", retriever_type=RetrieverType.HISTORICAL)
        self.collection = collection
        self.embedder = get_embedding_model()
        self.store = get_vector_store(collection)
        self._seed_events()

    def _seed_events(self):
        if not self.store.available:
            return
        try:
            existing = self.store.search(
                query_vector=self.embedder.embed_query("historical events"),
                limit=1,
            )
            if existing:
                return
        except Exception:
            pass
        for event in HISTORICAL_EVENTS:
            text = f"{event['name']}: {event['description']}"
            vec = self.embedder.embed([text])
            self.store.add(
                texts=[text],
                vectors=vec,
                metadata=[{
                    "name": event["name"],
                    "date": event["date"],
                    "entities": ",".join(event["entities"]),
                    "sectors": ",".join(event["sectors"]),
                    "regions": ",".join(event["regions"]),
                    "source": "historical_events",
                }],
            )

    def get_all_events(self) -> List[dict]:
        return HISTORICAL_EVENTS

    def get_events_by_sector(self, sector: str) -> List[dict]:
        sector_lower = sector.lower()
        return [e for e in HISTORICAL_EVENTS if sector_lower in [s.lower() for s in e.get("sectors", [])]]

    def get_events_by_region(self, region: str) -> List[dict]:
        region_lower = region.lower()
        return [e for e in HISTORICAL_EVENTS if region_lower in [r.lower() for r in e.get("regions", [])]]

    def get_events_by_entity(self, entity: str) -> List[dict]:
        entity_lower = entity.lower()
        return [e for e in HISTORICAL_EVENTS if any(entity_lower in ent.lower() for ent in e.get("entities", []))]

    def search_events(self, query: str, limit: int = 5) -> List[dict]:
        query_lower = query.lower()
        scored = []
        for event in HISTORICAL_EVENTS:
            score = 0.0
            search_text = f"{event['name']} {event['description']} {' '.join(event['entities'])} {' '.join(event['sectors'])} {' '.join(event['regions'])}".lower()
            for term in query_lower.split():
                if term in search_text:
                    score += 1.0
            entity_match = sum(1 for e in event["entities"] if e.lower() in query_lower)
            sector_match = sum(1 for s in event["sectors"] if s.lower() in query_lower)
            region_match = sum(1 for r in event["regions"] if r.lower() in query_lower)
            score += entity_match * 2 + sector_match * 1.5 + region_match * 1.5
            if score > 0:
                scored.append((score, event))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    async def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        results = []
        query_vec = self.embedder.embed_query(query)
        vector_results = self.store.search(query_vector=query_vec, limit=limit)
        seen_events = set()
        for r in vector_results:
            name = r.payload.get("name", "")
            if name and name not in seen_events:
                seen_events.add(name)
            results.append(
                RetrievalResult(
                    content=r.payload.get("text", r.payload.get("name", "")),
                    score=r.score,
                    source="historical_events",
                    retriever_type=RetrieverType.HISTORICAL,
                    metadata=r.payload,
                    id=r.id,
                )
            )
        keyword_events = self.search_events(query, limit)
        for event in keyword_events:
            name = event["name"]
            if name not in seen_events:
                seen_events.add(name)
                text = f"{event['name']}: {event['description']}"
                results.append(
                    RetrievalResult(
                        content=text,
                        score=0.7,
                        source="historical_events",
                        retriever_type=RetrieverType.HISTORICAL,
                        metadata=event,
                    )
                )
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
