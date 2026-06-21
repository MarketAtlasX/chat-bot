from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rag.graph_retrieval.neo4j_client import GraphDBClient


@dataclass
class GraphQueryResult:
    query: str
    source_entity: str
    target_entity: str
    paths: List[List[Tuple[str, str, str]]] = field(default_factory=list)
    context: str = ""
    raw_results: List[dict] = field(default_factory=list)


DEFAULT_GRAPH = {
    "Iran": [("supplies oil to", "China", "energy"), ("supplies oil to", "India", "energy"), ("supplies oil to", "Europe", "energy"), ("has tensions with", "USA", "geopolitical"), ("has tensions with", "Israel", "geopolitical"), ("supports", "Hamas", "geopolitical"), ("supports", "Hezbollah", "geopolitical"), ("threatens", "Hormuz Strait", "geopolitical")],
    "Russia": [("supplies gas to", "Europe", "energy"), ("supplies oil to", "China", "energy"), ("supplies oil to", "India", "energy"), ("has tensions with", "NATO", "geopolitical"), ("sanctioned by", "EU", "geopolitical"), ("sanctioned by", "USA", "geopolitical"), ("has conflict with", "Ukraine", "geopolitical")],
    "China": [("manufactures for", "USA", "trade"), ("manufactures for", "Europe", "trade"), ("competes with", "USA", "technology"), ("claims sovereignty over", "Taiwan", "geopolitical"), ("has tensions in", "South China Sea", "geopolitical"), ("imports oil from", "Russia", "energy"), ("imports oil from", "Iran", "energy"), ("imports oil from", "Saudi Arabia", "energy"), ("partnered with", "Russia", "geopolitical")],
    "USA": [("sanctions", "Iran", "geopolitical"), ("sanctions", "Russia", "geopolitical"), ("supports", "Taiwan", "geopolitical"), ("supports", "Israel", "geopolitical"), ("has military in", "Japan", "defense"), ("has military in", "South Korea", "defense"), ("competes with", "China", "technology"), ("imports from", "China", "trade"), ("exports LNG to", "Europe", "energy")],
    "Taiwan": [("manufactures", "semiconductors", "technology"), ("claimed by", "China", "geopolitical"), ("supplied by", "USA", "defense"), ("exports to", "Global", "technology")],
    "Saudi Arabia": [("supplies oil to", "Global", "energy"), ("leads", "OPEC", "energy"), ("has tensions with", "Iran", "geopolitical"), ("partnered with", "USA", "geopolitical")],
    "Europe": [("imports gas from", "Russia", "energy"), ("imports LNG from", "USA", "energy"), ("sanctions", "Russia", "geopolitical"), ("partnered with", "NATO", "defense")],
    "Ukraine": [("at war with", "Russia", "geopolitical"), ("supplied by", "NATO", "defense"), ("supplied by", "USA", "defense"), ("transits gas for", "Europe", "energy")],
    "Israel": [("has tensions with", "Iran", "geopolitical"), ("has conflict with", "Hamas", "geopolitical"), ("supported by", "USA", "defense")],
    "Hormuz Strait": [("transits", "global oil", "energy"), ("threatened by", "Iran", "geopolitical")],
    "Semiconductors": [("manufactured by", "Taiwan", "technology"), ("manufactured by", "South Korea", "technology"), ("essential for", "Global tech", "technology"), ("affected by", "geopolitical tensions", "geopolitical")],
    "OPEC": [("led by", "Saudi Arabia", "energy"), ("includes", "Iran", "energy"), ("controls", "global oil supply", "energy")],
    "NATO": [("includes", "USA", "defense"), ("includes", "Europe", "defense"), ("has tensions with", "Russia", "geopolitical"), ("supplies", "Ukraine", "defense")],
    "Hamas": [("supported by", "Iran", "geopolitical"), ("has conflict with", "Israel", "geopolitical")],
    "Hezbollah": [("supported by", "Iran", "geopolitical")],
}


class GraphQueryEngine:
    def __init__(self, neo4j_client: Optional[GraphDBClient] = None):
        self.neo4j = neo4j_client or GraphDBClient()

    def extract_entities(self, query: str) -> List[str]:
        query_lower = query.lower()
        found = []
        for entity in DEFAULT_GRAPH:
            if entity.lower() in query_lower:
                found.append(entity)
        if not found:
            for entity, relations in DEFAULT_GRAPH.items():
                for _, target, _ in relations:
                    if target.lower() in query_lower and target not in found:
                        found.append(target)
        return found

    def query_relationship(self, source: str, target: str, max_depth: int = 3) -> GraphQueryResult:
        paths = []
        visited = set()

        def dfs(current: str, target: str, path: list, depth: int):
            if depth > max_depth or current in visited:
                return
            if current == target and path:
                paths.append(list(path))
                return
            visited.add(current)
            if current in DEFAULT_GRAPH:
                for rel, neighbor, sector in DEFAULT_GRAPH[current]:
                    new_path = path + [(current, rel, neighbor)]
                    if neighbor == target:
                        paths.append(list(new_path))
                    else:
                        dfs(neighbor, target, new_path, depth + 1)
            visited.remove(current)

        dfs(source, target, [], 0)

        context_parts = []
        for p in paths[:5]:
            path_str = " -> ".join([f"{s}[{r}]→{t}" for s, r, t in p])
            context_parts.append(path_str)

        neo4j_results = []
        if self.neo4j.available:
            neo4j_results = self.neo4j.get_relations(source, depth=max_depth)
            for rel in neo4j_results:
                rel_source = rel.get("source", rel.get("name", ""))
                rel_target = rel.get("target", "")
                rel_type = rel.get("relation", rel.get("rel_type", ""))
                if rel_source and rel_target:
                    path_str = f"{rel_source}[{rel_type}]→{rel_target}"
                    if path_str not in context_parts:
                        context_parts.append(path_str)

        return GraphQueryResult(
            query=f"Relationship between {source} and {target}",
            source_entity=source,
            target_entity=target,
            paths=paths[:5],
            context="\n".join(context_parts),
            raw_results=neo4j_results,
        )

    def explore_entity(self, entity_name: str, depth: int = 2) -> GraphQueryResult:
        paths = []
        visited = set()

        def dfs(current: str, path: list, d: int):
            if d > depth or current in visited:
                return
            visited.add(current)
            if current in DEFAULT_GRAPH:
                for rel, neighbor, sector in DEFAULT_GRAPH[current]:
                    new_path = path + [(current, rel, neighbor)]
                    paths.append(new_path)
                    dfs(neighbor, new_path, d + 1)
            visited.remove(current)

        dfs(entity_name, [], 0)

        context_parts = []
        seen_paths = set()
        for p in paths:
            path_str = " -> ".join([f"{s}[{r}]→{t}" for s, r, t in p])
            if path_str not in seen_paths:
                seen_paths.add(path_str)
                context_parts.append(path_str)

        neo4j_results = []
        if self.neo4j.available:
            neo4j_results = self.neo4j.get_relations(entity_name, depth)
            for rel in neo4j_results:
                if "entities" in rel:
                    entities = rel["entities"]
                    rels = rel.get("relations", [])
                    for i in range(len(entities) - 1):
                        ps = f"{entities[i]}[{rels[i] if i < len(rels) else '?'}]→{entities[i+1]}"
                        if ps not in seen_paths:
                            seen_paths.add(ps)
                            context_parts.append(ps)
                else:
                    ps = f"{rel.get('source', entity_name)}[{rel.get('relation', rel.get('rel_type', '?'))}]→{rel.get('target', '?')}"
                    if ps not in seen_paths:
                        seen_paths.add(ps)
                        context_parts.append(ps)

        return GraphQueryResult(
            query=f"Explore entity: {entity_name}",
            source_entity=entity_name,
            target_entity="",
            paths=paths[:10],
            context="\n".join(context_parts),
            raw_results=neo4j_results,
        )
