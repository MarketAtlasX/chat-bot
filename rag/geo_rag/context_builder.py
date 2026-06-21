from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rag.retrievers.base import RetrievalResult, RetrieverType


@dataclass
class GeoContext:
    query: str
    news_context: str = ""
    historical_context: str = ""
    graph_context: str = ""
    market_context: str = ""
    combined_text: str = ""
    source_summary: str = ""
    metadata: dict = field(default_factory=dict)


class GeoContextBuilder:
    def build(
        self,
        query: str,
        news_results: Optional[List[RetrievalResult]] = None,
        historical_results: Optional[List[RetrievalResult]] = None,
        graph_results: Optional[List[RetrievalResult]] = None,
        market_results: Optional[List[RetrievalResult]] = None,
        max_tokens: int = 4000,
    ) -> GeoContext:
        news_context = self._format_results(news_results or [], "News Articles")
        historical_context = self._format_results(historical_results or [], "Historical Events")
        graph_context = self._format_results(graph_results or [], "Knowledge Graph")
        market_context = self._format_results(market_results or [], "Market Data")

        source_counts = {}
        for results, label in [
            (news_results or [], "news"),
            (historical_results or [], "historical"),
            (graph_results or [], "graph"),
            (market_results or [], "market"),
        ]:
            if results:
                source_counts[label] = len(results)

        source_summary = f"Retrieved from {sum(source_counts.values())} sources: " + ", ".join(
            f"{count} {name}" for name, count in source_counts.items()
        )

        sections = []
        if news_context:
            sections.append(news_context)
        if historical_context:
            sections.append(historical_context)
        if graph_context:
            sections.append(graph_context)
        if market_context:
            sections.append(market_context)

        combined = "\n\n".join(sections)

        if len(combined) > max_tokens:
            combined = self._truncate(combined, max_tokens)

        return GeoContext(
            query=query,
            news_context=news_context,
            historical_context=historical_context,
            graph_context=graph_context,
            market_context=market_context,
            combined_text=combined,
            source_summary=source_summary,
            metadata={"source_counts": source_counts},
        )

    def _format_results(self, results: List[RetrievalResult], section_title: str) -> str:
        if not results:
            return ""
        lines = [f"=== {section_title} ==="]
        for i, r in enumerate(results, 1):
            title = r.metadata.get("title", r.metadata.get("name", r.metadata.get("event_name", "")))
            source = r.metadata.get("source", r.source)
            lines.append(f"\n[{i}] {title} (relevance: {r.score:.2f})")
            if source:
                lines.append(f"    Source: {source}")
            content = r.content[:500]
            lines.append(f"    {content}")
        return "\n".join(lines)

    def build_minimal(self, query: str, source: str, results: list) -> str:
        header = f"=== {source} ===\n"
        items = []
        for i, r in enumerate(results[:3], 1):
            title = r.get("title") or r.get("name") or f"Result {i}"
            content = str(r.get("content", r.get("description", "")))[:200]
            items.append(f"[{i}] {title}\n    {content}")
        return header + "\n".join(items) if items else ""

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[Content truncated...]"
