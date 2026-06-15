import json
from typing import Any
from ..llm.ollama import get_llm
from ..knowledge.neo4j_client import Neo4jClient
from ..rag.retriever import retrieve_context


class GraphAgent:
    def __init__(self):
        self.llm = get_llm()
        self.neo4j = Neo4jClient()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        entities = self._extract_entities(query)

        all_relations = []
        for entity in entities[:5]:
            relations = self.neo4j.get_relations(entity, depth=2)
            all_relations.extend(relations)
            related = self.neo4j.query_related_entities(entity)
            all_relations.extend(related)

        knowledge = retrieve_context(query, limit=3)

        system_prompt = """You are a knowledge graph analyst. Use entity relationships to explain connections
between geopolitical events, entities, and market impacts. Think of the world as an interconnected network."""

        relations_text = ""
        if all_relations:
            relations_text = "Relationships found in knowledge graph:\n" + json.dumps(all_relations[:20], indent=2)
        elif entities:
            relations_text = f"Entities identified: {', '.join(entities)}"

        prompt = f"""Query: {query}

{relations_text}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

Explain the relationships and connections relevant to this query:"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        return {
            "agent": "GraphAgent",
            "response": response,
            "entities": entities,
            "relations_found": len(all_relations),
        }

    def _extract_entities(self, text: str) -> list[str]:
        prompt = f"""Extract all named entities (countries, companies, organizations, people, sectors, commodities) from this query.
Return ONLY a JSON array of strings, nothing else.
Query: {text}"""
        try:
            result = self.llm.generate(prompt, temperature=0.1)
            result = result.strip().strip("```json").strip("```").strip()
            return json.loads(result) if result.startswith("[") else []
        except Exception:
            return []
