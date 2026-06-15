import json
from typing import Any
from ..llm.ollama import get_llm
from ..rag.retriever import retrieve_context
from ..models import SimulationResult


class SimulationAgent:
    def __init__(self):
        self.llm = get_llm()

    def process(self, query: str, context: dict[str, Any] = None) -> dict[str, Any]:
        knowledge = retrieve_context(query, limit=4)

        system_prompt = """You are a geopolitical simulation specialist. Run scenario analysis to predict outcomes
of geopolitical events. Consider multiple variables, cascading effects, and probability distributions."""

        prompt = f"""Run a geopolitical simulation for this scenario.

Query: {query}

Relevant Knowledge:
{knowledge if knowledge else "No specific knowledge base results."}

Return the results in this EXACT JSON format:
{{
    "scenario": "Scenario description",
    "consequences": {{
        "Sector/Asset 1": "+X% or description",
        "Sector/Asset 2": "-Y% or description"
    }},
    "probability": 0.0-1.0,
    "time_horizon": "short/medium/long term",
    "key_risks": ["risk1", "risk2", "risk3"]
}}

Return ONLY the JSON, no other text."""

        try:
            result = self.llm.generate(prompt, system_prompt=system_prompt)
            result = result.strip().strip("```json").strip("```").strip()
            sim_data = json.loads(result)

            simulation = SimulationResult(**sim_data)

            response = f"""## Simulation: {simulation.scenario}

### Expected Consequences
"""
            for sector, impact in simulation.consequences.items():
                response += f"- **{sector}**: {impact}\n"

            response += f"""
### Probability
{simulation.probability:.0%}

### Time Horizon
{simulation.time_horizon}

### Key Risks
"""
            for risk in simulation.key_risks:
                response += f"- {risk}\n"

        except Exception:
            response = f"""## Scenario Analysis

Unable to generate structured simulation. Performing scenario analysis...

Query: {query}"""
            fallback = self.llm.generate(
                f"Analyze this geopolitical what-if scenario in detail:\n{query}\n\nContext:\n{knowledge}",
                system_prompt
            )
            response += f"\n\n{fallback}"
            sim_data = {}

        return {
            "agent": "SimulationAgent",
            "response": response,
            "simulation_data": sim_data,
        }
