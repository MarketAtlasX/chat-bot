from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraphDBClient:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri
        self.user = user or "neo4j"
        self.password = password or "test"
        self._driver = None
        self._available = False
        self._connect()

    def _connect(self):
        if not self.uri:
            logger.info("No Neo4j URI configured, using fallback")
            return
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self._driver.verify_connectivity()
            self._available = True
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def query(self, cypher: str, params: Optional[dict] = None) -> List[dict]:
        if not self._available:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return []

    def merge_entity(self, name: str, entity_type: str, properties: Optional[dict] = None) -> bool:
        if not self._available:
            return False
        try:
            with self._driver.session() as session:
                session.run(
                    "MERGE (e:Entity {name: $name}) SET e.type = $entity_type, e += $props",
                    name=name,
                    entity_type=entity_type,
                    props=properties or {},
                )
                return True
        except Exception as e:
            logger.error(f"Failed to merge entity: {e}")
            return False

    def merge_relation(self, source: str, target: str, relation: str, properties: Optional[dict] = None) -> bool:
        if not self._available:
            return False
        try:
            with self._driver.session() as session:
                session.run(
                    "MATCH (s:Entity {name: $source}), (t:Entity {name: $target}) "
                    "MERGE (s)-[r:RELATES {type: $relation}]->(t) "
                    "SET r += $props",
                    source=source,
                    target=target,
                    relation=relation,
                    props=properties or {},
                )
                return True
        except Exception as e:
            logger.error(f"Failed to merge relation: {e}")
            return False

    def get_entity(self, name: str) -> Optional[dict]:
        results = self.query(
            "MATCH (e:Entity {name: $name}) RETURN e.name as name, e.type as type, e",
            {"name": name},
        )
        if results:
            return results[0]
        return None

    def get_relations(self, entity_name: str, depth: int = 1) -> List[dict]:
        if depth <= 1:
            return self.query(
                "MATCH (e:Entity {name: $name})-[r]->(t:Entity) "
                "RETURN e.name as source, r.type as relation, t.name as target, type(r) as rel_type",
                {"name": entity_name},
            )
        return self.query(
            "MATCH path = (e:Entity {name: $name})-[*1..$depth]->(t:Entity) "
            "RETURN [n in nodes(path) | n.name] as entities, "
            "[r in relationships(path) | type(r)] as relations",
            {"name": entity_name, "depth": depth},
        )

    def query_related_entities(self, entity_name: str, relation_type: str) -> List[dict]:
        return self.query(
            "MATCH (e:Entity {name: $name})-[r {type: $relation_type}]->(t:Entity) "
            "RETURN t.name as name, t.type as type",
            {"name": entity_name, "relation_type": relation_type},
        )

    def get_graph_context(self, entity_name: str, depth: int = 2) -> str:
        relations = self.get_relations(entity_name, depth)
        if not relations:
            return ""
        lines = []
        for rel in relations[:20]:
            if "entities" in rel:
                entities = rel["entities"]
                rels = rel["relations"]
                path_str = " -> ".join(
                    [f"{entities[i]}[{rels[i] if i < len(rels) else '?'}]→{entities[i+1] if i+1 < len(entities) else '?'}"
                     for i in range(len(entities) - 1)]
                )
                lines.append(f"  Path: {path_str}")
            else:
                lines.append(f"  {rel.get('source', '?')} -[{rel.get('relation', rel.get('rel_type', '?'))}]→ {rel.get('target', '?')}")
        return "Graph Context:\n" + "\n".join(lines) if lines else ""

    def close(self):
        if self._driver:
            self._driver.close()
