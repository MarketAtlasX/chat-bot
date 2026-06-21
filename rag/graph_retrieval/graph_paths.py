from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rag.graph_retrieval.graph_query import GraphQueryEngine, GraphQueryResult


@dataclass
class GraphPath:
    entities: List[str]
    relations: List[str]
    sectors: List[str]
    path_string: str
    depth: int


class GraphPathExtractor:
    def __init__(self, engine: Optional[GraphQueryEngine] = None):
        self.engine = engine or GraphQueryEngine()

    def extract_paths(self, query: str, max_paths: int = 5) -> List[GraphPath]:
        entities = self.engine.extract_entities(query)

        if len(entities) >= 2:
            results = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    result = self.engine.query_relationship(entities[i], entities[j])
                    results.append(result)
            for entity in entities[:3]:
                result = self.engine.explore_entity(entity)
                results.append(result)
        else:
            entity = entities[0] if entities else "Global"
            results = [self.engine.explore_entity(entity)]

        paths = self._parse_results(results)
        paths.sort(key=lambda p: (len(p.entities), len(p.relations)), reverse=True)
        return paths[:max_paths]

    def _parse_results(self, results: List[GraphQueryResult]) -> List[GraphPath]:
        graph_paths = []
        seen = set()
        for result in results:
            for p in result.paths:
                entities = []
                relations = []
                sectors = []
                for step in p:
                    source, rel, target = step
                    if not entities or entities[-1] != source:
                        entities.append(source)
                    relations.append(rel)
                    entities.append(target)
                    rel_to_sector = {
                        "supplies oil to": "energy",
                        "supplies gas to": "energy",
                        "exports LNG to": "energy",
                        "transits gas for": "energy",
                        "transits": "energy",
                        "leads": "energy",
                        "includes": "energy",
                        "controls": "energy",
                        "manufactures for": "trade",
                        "imports from": "trade",
                        "imports oil from": "energy",
                        "exports to": "trade",
                        "manufactures": "technology",
                        "essential for": "technology",
                        "manufactured by": "technology",
                        "has tensions with": "geopolitical",
                        "sanctions": "geopolitical",
                        "sanctioned by": "geopolitical",
                        "supports": "geopolitical",
                        "supplied by": "defense",
                        "claimed by": "geopolitical",
                        "has conflict with": "geopolitical",
                        "at war with": "geopolitical",
                        "competes with": "geopolitical",
                        "partnered with": "geopolitical",
                        "has military in": "defense",
                        "threatens": "geopolitical",
                        "threatened by": "geopolitical",
                        "affected by": "geopolitical",
                    }
                    sectors.append(rel_to_sector.get(rel, "general"))
                path_str = " -> ".join(
                    [f"{entities[i]}[{relations[i]}]→{entities[i+1]}" for i in range(len(relations))]
                )
                if path_str not in seen:
                    seen.add(path_str)
                    graph_paths.append(
                        GraphPath(
                            entities=list(dict.fromkeys(entities)),
                            relations=relations,
                            sectors=list(dict.fromkeys(sectors)),
                            path_string=path_str,
                            depth=len(relations),
                        )
                    )
        return graph_paths

    def summarize_paths(self, paths: List[GraphPath]) -> str:
        if not paths:
            return "No graph paths found."
        lines = []
        for i, path in enumerate(paths, 1):
            lines.append(f"{i}. {path.path_string}")
            lines.append(f"   Sectors: {', '.join(path.sectors)}")
            lines.append(f"   Entities: {', '.join(path.entities)}")
        return "\n".join(lines)

    def paths_to_context(self, paths: List[GraphPath]) -> str:
        if not paths:
            return ""
        lines = ["Knowledge Graph Context:"]
        for path in paths:
            lines.append(f"  - {path.path_string}")
            lines.append(f"    (sectors: {', '.join(path.sectors)})")
        return "\n".join(lines)
