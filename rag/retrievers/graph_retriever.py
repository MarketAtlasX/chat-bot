from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from rag.retrievers.base import BaseRetriever, RetrievalResult, RetrieverType

logger = logging.getLogger(__name__)


DEFAULT_GRAPH_DATA = {
    "Iran": {
        "type": "country",
        "relationships": [
            ("supplies oil to", "China", "energy"),
            ("supplies oil to", "India", "energy"),
            ("supplies oil to", "Europe", "energy"),
            ("has tensions with", "USA", "geopolitical"),
            ("has tensions with", "Israel", "geopolitical"),
            ("supports", "Hamas", "geopolitical"),
            ("supports", "Hezbollah", "geopolitical"),
            ("threatens", "Hormuz Strait", "geopolitical"),
        ],
    },
    "Russia": {
        "type": "country",
        "relationships": [
            ("supplies gas to", "Europe", "energy"),
            ("supplies oil to", "China", "energy"),
            ("supplies oil to", "India", "energy"),
            ("has tensions with", "NATO", "geopolitical"),
            ("sanctioned by", "EU", "geopolitical"),
            ("sanctioned by", "USA", "geopolitical"),
            ("has conflict with", "Ukraine", "geopolitical"),
        ],
    },
    "China": {
        "type": "country",
        "relationships": [
            ("manufactures for", "USA", "trade"),
            ("manufactures for", "Europe", "trade"),
            ("competes with", "USA", "technology"),
            ("claims sovereignty over", "Taiwan", "geopolitical"),
            ("has tensions in", "South China Sea", "geopolitical"),
            ("imports oil from", "Russia", "energy"),
            ("imports oil from", "Iran", "energy"),
            ("imports oil from", "Saudi Arabia", "energy"),
            ("invests in", "Africa", "trade"),
            ("partnered with", "Russia", "geopolitical"),
        ],
    },
    "USA": {
        "type": "country",
        "relationships": [
            ("sanctions", "Iran", "geopolitical"),
            ("sanctions", "Russia", "geopolitical"),
            ("supports", "Taiwan", "geopolitical"),
            ("supports", "Israel", "geopolitical"),
            ("has military in", "Japan", "defense"),
            ("has military in", "South Korea", "defense"),
            ("competes with", "China", "technology"),
            ("imports from", "China", "trade"),
            ("exports", "LNG to", "Europe", "energy"),
        ],
    },
    "Taiwan": {
        "type": "country",
        "relationships": [
            ("manufactures", "semiconductors", "technology"),
            ("claimed by", "China", "geopolitical"),
            ("supplied by", "USA", "defense"),
            ("exports to", "Global", "technology"),
        ],
    },
    "Saudi Arabia": {
        "type": "country",
        "relationships": [
            ("supplies oil to", "Global", "energy"),
            ("leads", "OPEC", "energy"),
            ("has tensions with", "Iran", "geopolitical"),
            ("partnered with", "USA", "geopolitical"),
        ],
    },
    "Europe": {
        "type": "region",
        "relationships": [
            ("imports gas from", "Russia", "energy"),
            ("imports LNG from", "USA", "energy"),
            ("sanctions", "Russia", "geopolitical"),
            ("partnered with", "NATO", "defense"),
        ],
    },
    "Ukraine": {
        "type": "country",
        "relationships": [
            ("at war with", "Russia", "geopolitical"),
            ("supplied by", "NATO", "defense"),
            ("supplied by", "USA", "defense"),
            ("transits gas for", "Europe", "energy"),
        ],
    },
    "Israel": {
        "type": "country",
        "relationships": [
            ("has tensions with", "Iran", "geopolitical"),
            ("has conflict with", "Hamas", "geopolitical"),
            ("supported by", "USA", "defense"),
            ("discovers gas in", "Eastern Mediterranean", "energy"),
        ],
    },
    "Hormuz Strait": {
        "type": "chokepoint",
        "relationships": [
            ("transits", "global oil", "energy"),
            ("threatened by", "Iran", "geopolitical"),
        ],
    },
    "Semiconductors": {
        "type": "industry",
        "relationships": [
            ("manufactured by", "Taiwan", "technology"),
            ("manufactured by", "South Korea", "technology"),
            ("essential for", "Global tech", "technology"),
            ("affected by", "geopolitical tensions", "geopolitical"),
        ],
    },
    "OPEC": {
        "type": "organization",
        "relationships": [
            ("led by", "Saudi Arabia", "energy"),
            ("includes", "Iran", "energy"),
            ("controls", "global oil supply", "energy"),
        ],
    },
    "NATO": {
        "type": "organization",
        "relationships": [
            ("includes", "USA", "defense"),
            ("includes", "Europe", "defense"),
            ("has tensions with", "Russia", "geopolitical"),
            ("supplies", "Ukraine", "defense"),
        ],
    },
}


class GraphRetriever(BaseRetriever):
    def __init__(self, neo4j_uri: Optional[str] = None):
        super().__init__(name="graph_retriever", retriever_type=RetrieverType.GRAPH)
        self.neo4j_uri = neo4j_uri
        self._neo4j_driver = None
        self._try_connect_neo4j()

    def _try_connect_neo4j(self):
        if not self.neo4j_uri:
            return
        try:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(self.neo4j_uri)
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Using in-memory graph.")

    def _find_entities(self, query: str) -> List[str]:
        query_lower = query.lower()
        found = []
        for entity in DEFAULT_GRAPH_DATA:
            if entity.lower() in query_lower:
                found.append(entity)
        for entity in DEFAULT_GRAPH_DATA:
            data = DEFAULT_GRAPH_DATA[entity]
            if not isinstance(data, dict):
                continue
            for rel in data.get("relationships", []):
                if not isinstance(rel, (list, tuple)) or len(rel) < 2:
                    continue
                target = rel[1]
                if isinstance(target, str) and target.lower() in query_lower and target not in found:
                    found.append(target)
        return found if found else list(DEFAULT_GRAPH_DATA.keys())[:3]

    def _extract_paths(self, query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        entities = self._find_entities(query)
        paths = []
        visited = set()
        def dfs(current: str, path: List[tuple], depth: int):
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            if current in DEFAULT_GRAPH_DATA:
                data = DEFAULT_GRAPH_DATA[current]
                relations = data.get("relationships", []) if isinstance(data, dict) else []
                for rel_entry in relations:
                    if not isinstance(rel_entry, (list, tuple)) or len(rel_entry) < 3:
                        continue
                    rel, target, sector = rel_entry[0], rel_entry[1], rel_entry[2]
                    new_path = path + [(current, rel, target, sector)]
                    paths.append({
                        "path": new_path,
                        "entities": list(dict.fromkeys([e[0] for e in new_path] + [e[2] for e in new_path])),
                        "sectors": list(dict.fromkeys([e[3] for e in new_path])),
                        "depth": depth + 1,
                    })
                    dfs(target, new_path, depth + 1)
                    if target in DEFAULT_GRAPH_DATA and isinstance(DEFAULT_GRAPH_DATA[target], dict):
                        for re2 in DEFAULT_GRAPH_DATA[target].get("relationships", []):
                            if not isinstance(re2, (list, tuple)) or len(re2) < 3:
                                continue
                            rel2, target2, sector2 = re2[0], re2[1], re2[2]
                            new_path2 = new_path + [(target, rel2, target2, sector2)]
                            paths.append({
                                "path": new_path2,
                                "entities": list(dict.fromkeys([e[0] for e in new_path2] + [e[2] for e in new_path2])),
                                "sectors": list(dict.fromkeys([e[3] for e in new_path2])),
                                "depth": depth + 2,
                            })
            visited.remove(current)
        for entity in entities:
            dfs(entity, [], 0)
        paths.sort(key=lambda p: (len(p["entities"]), len(p["path"])), reverse=True)
        return paths[:10]

    async def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[RetrievalResult]:
        paths = self._extract_paths(query)
        results = []
        for i, p in enumerate(paths):
            path_str = " -> ".join([f"{s}[{r}]→{t}" for s, r, t, _ in p["path"]])
            sector_str = ", ".join(p["sectors"])
            entity_list = ", ".join(p["entities"])
            content = f"Graph Path: {path_str}\nEntities: {entity_list}\nSectors: {sector_str}"
            score = 1.0 - (p["depth"] * 0.1)
            results.append(
                RetrievalResult(
                    content=content,
                    score=max(score, 0.3),
                    source="knowledge_graph",
                    retriever_type=RetrieverType.GRAPH,
                    metadata={
                        "entities": entity_list,
                        "sectors": sector_str,
                        "depth": p["depth"],
                        "path": path_str,
                    },
                )
            )
        if self._neo4j_driver:
            try:
                with self._neo4j_driver.session() as session:
                    entities = self._find_entities(query)
                    for entity in entities[:3]:
                        result = session.run(
                            "MATCH (e)-[r]->(t) WHERE e.name = $name RETURN e.name as source, type(r) as rel, t.name as target",
                            name=entity,
                        )
                        for record in result:
                            content = f"Graph: {record['source']} -[{record['rel']}]-> {record['target']}"
                            results.append(
                                RetrievalResult(
                                    content=content,
                                    score=0.8,
                                    source="neo4j",
                                    retriever_type=RetrieverType.GRAPH,
                                    metadata={
                                        "source": record["source"],
                                        "relation": record["rel"],
                                        "target": record["target"],
                                    },
                                )
                            )
            except Exception as e:
                logger.warning(f"Neo4j query failed: {e}")
        dedup = {}
        for r in results:
            if r.content not in dedup:
                dedup[r.content] = r
        return list(dedup.values())[:limit]
