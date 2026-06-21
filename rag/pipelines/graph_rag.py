from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from rag.graph_retrieval.graph_paths import GraphPathExtractor
from rag.graph_retrieval.graph_query import GraphQueryEngine
from rag.graph_retrieval.neo4j_client import GraphDBClient
from rag.retrievers.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGResult:
    query: str
    paths: list = field(default_factory=list)
    context: str = ""
    path_summary: str = ""
    entities_found: List[str] = field(default_factory=list)


class GraphRAGPipeline:
    def __init__(self, neo4j_uri: Optional[str] = None):
        self.neo4j_client = GraphDBClient(uri=neo4j_uri)
        self.query_engine = GraphQueryEngine(neo4j_client=self.neo4j_client)
        self.path_extractor = GraphPathExtractor(engine=self.query_engine)
        self.graph_retriever = GraphRetriever(neo4j_uri=neo4j_uri)

    def query_relationship(self, source: str, target: str) -> GraphRAGResult:
        result = self.query_engine.query_relationship(source, target)
        paths = [
            {
                "path": [{"source": s, "relation": r, "target": t} for s, r, t in p],
                "path_string": " -> ".join([f"{s}[{r}]→{t}" for s, r, t in p]),
            }
            for p in result.paths
        ]
        return GraphRAGResult(
            query=f"Relationship: {source} -> {target}",
            paths=paths,
            context=result.context,
            path_summary=result.context[:500] if result.context else "No relationship paths found.",
            entities_found=[source, target],
        )

    def explore_entity(self, entity: str) -> GraphRAGResult:
        result = self.query_engine.explore_entity(entity)
        paths = [
            {
                "path": [{"source": s, "relation": r, "target": t} for s, r, t in p],
                "path_string": " -> ".join([f"{s}[{r}]→{t}" for s, r, t in p]),
            }
            for p in result.paths
        ]
        return GraphRAGResult(
            query=f"Explore: {entity}",
            paths=paths,
            context=result.context,
            path_summary=result.context[:500] if result.context else "No connections found.",
            entities_found=[entity],
        )

    async def query(
        self,
        question: str,
        limit: int = 5,
    ) -> GraphRAGResult:
        graph_paths = self.path_extractor.extract_paths(question, max_paths=limit)
        path_list = [
            {
                "entities": gp.entities,
                "relations": gp.relations,
                "sectors": gp.sectors,
                "path_string": gp.path_string,
                "depth": gp.depth,
            }
            for gp in graph_paths
        ]
        entities = self.query_engine.extract_entities(question)
        context = self.path_extractor.paths_to_context(graph_paths)
        summary = self.path_extractor.summarize_paths(graph_paths)
        return GraphRAGResult(
            query=question,
            paths=path_list,
            context=context,
            path_summary=summary,
            entities_found=entities,
        )
