import json
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context
from ..knowledge.neo4j_client import Neo4jClient


class NewsAgent:
    def __init__(self):
        self.llm = get_llm()
        self.neo4j = Neo4jClient()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=3)

        system_prompt = """You are a geopolitical news analyst. Summarize recent events relevant to the query.
Be factual, precise, and cite sources where possible. Focus on actionable intelligence."""

        prompt = f"""Query: {query}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

{'Conversation Context: ' + context.get('conversation_context', '') if context and context.get('conversation_context') else ''}

Provide a concise summary of relevant news/events:"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        graph_context = ""
        entities = self._extract_entities(query)
        for entity in entities:
            gc = self.neo4j.get_graph_context(entity)
            if gc:
                graph_context += f"\nRelations for {entity}:\n{gc}"

        return {
            "agent": "NewsAgent",
            "response": response,
            "sources": ["News API", "Knowledge Base"],
            "entities": entities,
            "graph_context": graph_context,
        }

    def _extract_entities(self, text: str) -> list[str]:
        prompt = f"""Extract geopolitical entities (countries, organizations, people, sectors) from this query.
Return ONLY a JSON array of strings, nothing else.
Query: {text}"""
        try:
            result = self.llm.generate(prompt, temperature=0.1)
            result = result.strip().strip("```json").strip("```").strip()
            return json.loads(result) if result.startswith("[") else []
        except Exception:
            return []
