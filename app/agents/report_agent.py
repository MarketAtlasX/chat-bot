import json
from datetime import datetime
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context
from ..models import IntelligenceReport


class ReportAgent:
    def __init__(self):
        self.llm = get_llm()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=6)

        system_prompt = """You are a senior intelligence analyst generating structured geopolitical intelligence reports.
Produce professional, comprehensive reports following intelligence community standards."""

        prompt = f"""Generate a MarketAtlas Intelligence Report for this query.

Query: {query}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

{'Agent Inputs:' if context else ''}
{json.dumps({k: v for k, v in (context or {}).items() if k != 'conversation_context'}, indent=2)[:2000] if context else ''}

Return the report in this EXACT JSON format:
{{
    "title": "Report title",
    "event": "Event description",
    "affected_sectors": ["sector1", "sector2"],
    "risk_score": 0.0-1.0,
    "expected_market_impact": "Description of expected market impact",
    "recommended_assets": ["TICKER1", "TICKER2"],
    "confidence": 0.0-1.0,
    "reasoning": "Detailed reasoning",
    "sources": ["source1", "source2"]
}}

Return ONLY the JSON, no other text."""

        try:
            result = self.llm.generate(prompt, system_prompt=system_prompt)
            result = result.strip().strip("```json").strip("```").strip()
            report_data = json.loads(result)

            report_data["timestamp"] = datetime.utcnow().isoformat()

            report = IntelligenceReport(**report_data)

            report_text = f"""# MarketAtlas Intelligence Report

## {report.title}

### Event
{report.event}

### Affected Sectors
{', '.join(report.affected_sectors)}

### Risk Score
{report.risk_score:.2f}

### Expected Market Impact
{report.expected_market_impact}

### Recommended Assets
{', '.join(report.recommended_assets)}

### Confidence
{report.confidence:.2f}

### Reasoning
{report.reasoning}

### Sources
{chr(10).join('- ' + s for s in report.sources) if report.sources else 'N/A'}

### Generated
{report.timestamp}
"""

        except Exception:
            report_text = f"# Intelligence Report\n\n## Analysis for: {query}\n\nComprehensive analysis could not be generated in structured format.\n\nGenerating free-form analysis instead..."
            fallback = self.llm.generate(
                f"Provide a comprehensive intelligence report analyzing: {query}\n\nKnowledge context:\n{knowledge}",
                system_prompt
            )
            report_text += f"\n\n{fallback}"

        return {
            "agent": "ReportAgent",
            "response": report_text,
            "report_data": report_data if 'report_data' in locals() else None,
        }
