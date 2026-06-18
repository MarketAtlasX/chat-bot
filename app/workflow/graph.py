import uuid
from typing import Any, Literal
from langgraph.graph import StateGraph, END
from ..models import IntentType, ChatResponse
from ..agents import (
    IntentRouter,
    NewsAgent,
    MarketAgent,
    ImpactAgent,
    GraphAgent,
    ForecastAgent,
    RecommendationAgent,
    ReportAgent,
    SimulationAgent,
    DebateAgent,
    EventSimilarityAgent,
)
from ..memory.short_term import short_term_memory
from ..rag.retriever import seed_knowledge_base


class AgentState(dict):
    query: str
    conversation_id: str
    user_id: str
    intent: IntentType
    intent_confidence: float
    agents_used: list[str]
    sources: list[str]
    agent_responses: dict[str, Any]
    final_response: str
    confidence: float
    error: str
    _context: dict[str, Any]


router = IntentRouter()
news_agent = NewsAgent()
market_agent = MarketAgent()
impact_agent = ImpactAgent()
graph_agent = GraphAgent()
forecast_agent = ForecastAgent()
recommendation_agent = RecommendationAgent()
report_agent = ReportAgent()
simulation_agent = SimulationAgent()
debate_agent = DebateAgent()
event_similarity_agent = EventSimilarityAgent()


def route_intent(state: AgentState) -> AgentState:
    intent, confidence = router.classify(state["query"])
    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["agents_used"] = router.get_agents_for_intent(intent)
    state["agent_responses"] = {}
    state["sources"] = []

    history = short_term_memory.format_context(state["conversation_id"])
    context = {"conversation_context": history}

    state["_context"] = context
    return state


def decide_agents(state: AgentState) -> Literal["debate", "report", "execute_debate", "execute_report", "execute_direct", "execute_news", "execute_market", "execute_impact", "execute_graph", "execute_forecast", "execute_recommendation", "execute_simulation", "execute_similarity", "execute_similarity_pipeline"]:
    intent = state["intent"]

    if intent == IntentType.REPORT:
        return "execute_report"
    if intent == IntentType.SIMULATION:
        return "execute_simulation"
    if intent == IntentType.SIMILARITY:
        return "execute_similarity_pipeline"

    if intent == IntentType.NEWS:
        return "execute_news"
    if intent == IntentType.MARKET:
        return "execute_market"
    if intent == IntentType.IMPACT:
        return "execute_impact"
    if intent == IntentType.GRAPH:
        return "execute_graph"
    if intent == IntentType.RECOMMENDATION:
        return "execute_recommendation"

    agents = state["agents_used"]
    if len(agents) > 2:
        return "debate"
    return "execute_direct"


def _ensure_context(state: AgentState) -> None:
    if "_context" not in state:
        state["_context"] = {}
    if "agent_responses" not in state:
        state["agent_responses"] = {}
    if "sources" not in state:
        state["sources"] = []


def execute_news(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = news_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["NewsAgent"] = result["response"]
    state["sources"].extend(result.get("sources", []))
    state["final_response"] = result["response"]
    return state


def execute_market(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = market_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["MarketAgent"] = result["response"]
    if "market_data" in result:
        state["_context"]["market_data"] = result["market_data"]
    if "explanations" in result:
        state["_context"]["explanations"] = state["_context"].get("explanations", {})
        state["_context"]["explanations"]["MarketAgent"] = result["explanations"]
    if "explanation_text" in result:
        state["_context"]["explanation_texts"] = state["_context"].get("explanation_texts", {})
        state["_context"]["explanation_texts"]["MarketAgent"] = result["explanation_text"]
    state["final_response"] = result["response"]
    return state


def execute_impact(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = impact_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["ImpactAgent"] = result["response"]
    state["_context"]["impact_analysis"] = result["response"]
    state["sources"].extend(result.get("sources", []))
    if "explanations" in result:
        state["_context"]["explanations"] = state["_context"].get("explanations", {})
        state["_context"]["explanations"]["ImpactAgent"] = result["explanations"]
    if "explanation_text" in result:
        state["_context"]["explanation_texts"] = state["_context"].get("explanation_texts", {})
        state["_context"]["explanation_texts"]["ImpactAgent"] = result["explanation_text"]
    state["final_response"] = result["response"]
    return state


def execute_graph(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = graph_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["GraphAgent"] = result["response"]
    state["sources"].extend(result.get("sources", []))
    state["final_response"] = result["response"]
    return state


def execute_forecast(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = forecast_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["ForecastAgent"] = result["response"]
    if "explanations" in result:
        state["_context"]["explanations"] = state["_context"].get("explanations", {})
        state["_context"]["explanations"]["ForecastAgent"] = result["explanations"]
    if "explanation_text" in result:
        state["_context"]["explanation_texts"] = state["_context"].get("explanation_texts", {})
        state["_context"]["explanation_texts"]["ForecastAgent"] = result["explanation_text"]
    state["final_response"] = result["response"]
    return state


def execute_recommendation(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = recommendation_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["RecommendationAgent"] = result["response"]
    state["final_response"] = result["response"]
    return state


def execute_simulation(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = simulation_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["SimulationAgent"] = result["response"]
    state["final_response"] = result["response"]
    return state


def execute_report(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = report_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["ReportAgent"] = result["response"]
    state["final_response"] = result["response"]
    state["sources"].extend(result.get("sources", []))
    return state


def execute_similarity(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = event_similarity_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["EventSimilarityAgent"] = result["response"]
    state["final_response"] = result["response"]
    state["sources"].append("Event Memory Database")
    return state


def execute_similarity_pipeline(state: AgentState) -> AgentState:
    _ensure_context(state)

    news_result = news_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["NewsAgent"] = news_result["response"]
    state["sources"].extend(news_result.get("sources", []))

    sim_result = event_similarity_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["EventSimilarityAgent"] = sim_result["response"]
    state["sources"].append("Event Memory Database")
    sim_data = sim_result.get("similarity_data", {})
    sim_expl = sim_result.get("explanation_text", "")

    impact_result = impact_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["ImpactAgent"] = impact_result["response"]
    state["_context"]["impact_analysis"] = impact_result["response"]
    state["sources"].extend(impact_result.get("sources", []))
    impact_expl = impact_result.get("explanation_text", "")

    forecast_result = forecast_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["ForecastAgent"] = forecast_result["response"]
    report_result = report_agent.process(state["query"], state.get("_context"))
    state["agent_responses"]["ReportAgent"] = report_result["response"]
    state["sources"].extend(report_result.get("sources", []))

    state["_context"]["similarity_data"] = sim_data
    combined_expl = sim_expl
    if impact_expl:
        combined_expl = combined_expl + "\n\n" + impact_expl if combined_expl else impact_expl
    state["final_response"] = event_similarity_agent.format_full_report(
        state["query"], sim_data,
        news_result["response"],
        impact_result["response"],
        forecast_result["response"],
        report_result["response"],
        explanation_text=combined_expl,
    )

    state["agents_used"] = [
        "NewsAgent", "EventSimilarityAgent", "ImpactAgent",
        "ForecastAgent", "ReportAgent",
    ]
    return state


def execute_debate(state: AgentState) -> AgentState:
    _ensure_context(state)
    result = debate_agent.run_debate(state["query"], state.get("_context"))
    state["agent_responses"]["DebateAgent"] = result["response"]
    state["agent_responses"]["Perspectives"] = result.get("perspectives", [])
    state["final_response"] = result["response"]
    return state


def execute_direct(state: AgentState) -> AgentState:
    _ensure_context(state)
    for agent_name in state["agents_used"]:
        if agent_name == "NewsAgent":
            r = news_agent.process(state["query"], state.get("_context"))
        elif agent_name == "MarketAgent":
            r = market_agent.process(state["query"], state.get("_context"))
        elif agent_name == "ImpactAgent":
            r = impact_agent.process(state["query"], state.get("_context"))
        elif agent_name == "GraphAgent":
            r = graph_agent.process(state["query"], state.get("_context"))
        elif agent_name == "ForecastAgent":
            r = forecast_agent.process(state["query"], state.get("_context"))
        elif agent_name == "RecommendationAgent":
            r = recommendation_agent.process(state["query"], state.get("_context"))
        elif agent_name == "SimulationAgent":
            r = simulation_agent.process(state["query"], state.get("_context"))
        elif agent_name == "ReportAgent":
            r = report_agent.process(state["query"], state.get("_context"))
        elif agent_name == "EventSimilarityAgent":
            r = event_similarity_agent.process(state["query"], state.get("_context"))
        else:
            continue
        state["agent_responses"][agent_name] = r["response"]
        if "sources" in r:
            state["sources"].extend(r["sources"])

    combined = "\n\n".join([f"### {name}\n{resp}" for name, resp in state["agent_responses"].items()])
    state["final_response"] = combined
    return state


def calculate_confidence(state: AgentState) -> AgentState:
    base = state["intent_confidence"]
    num_responses = len(state["agent_responses"])
    response_bonus = min(num_responses * 0.05, 0.2)
    state["confidence"] = min(base + response_bonus, 0.95)
    return state


def store_memory(state: AgentState) -> AgentState:
    short_term_memory.add_turn(state["conversation_id"], "user", state["query"])
    short_term_memory.add_turn(state["conversation_id"], "assistant", state["final_response"])
    return state


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("route_intent", route_intent)
    workflow.add_node("execute_news", execute_news)
    workflow.add_node("execute_market", execute_market)
    workflow.add_node("execute_impact", execute_impact)
    workflow.add_node("execute_graph", execute_graph)
    workflow.add_node("execute_forecast", execute_forecast)
    workflow.add_node("execute_recommendation", execute_recommendation)
    workflow.add_node("execute_simulation", execute_simulation)
    workflow.add_node("execute_report", execute_report)
    workflow.add_node("execute_similarity", execute_similarity)
    workflow.add_node("execute_similarity_pipeline", execute_similarity_pipeline)
    workflow.add_node("execute_debate", execute_debate)
    workflow.add_node("execute_direct", execute_direct)
    workflow.add_node("calculate_confidence", calculate_confidence)
    workflow.add_node("store_memory", store_memory)

    workflow.set_entry_point("route_intent")

    workflow.add_conditional_edges(
        "route_intent",
        decide_agents,
        {
            "debate": "execute_debate",
            "report": "execute_report",
            "execute_debate": "execute_debate",
            "execute_report": "execute_report",
            "execute_similarity": "execute_similarity",
            "execute_similarity_pipeline": "execute_similarity_pipeline",
            "execute_direct": "execute_direct",
            "execute_news": "execute_news",
            "execute_market": "execute_market",
            "execute_impact": "execute_impact",
            "execute_graph": "execute_graph",
            "execute_forecast": "execute_forecast",
            "execute_recommendation": "execute_recommendation",
            "execute_simulation": "execute_simulation",
        }
    )

    execution_nodes = [
        "execute_news", "execute_market", "execute_impact", "execute_graph",
        "execute_forecast", "execute_recommendation", "execute_simulation",
        "execute_report", "execute_similarity", "execute_similarity_pipeline", "execute_debate", "execute_direct",
    ]
    for node in execution_nodes:
        workflow.add_edge(node, "calculate_confidence")

    workflow.add_edge("calculate_confidence", "store_memory")
    workflow.add_edge("store_memory", END)

    return workflow.compile()


graph = build_workflow()


async def run_chat(query: str, conversation_id: str = None, user_id: str = "default") -> ChatResponse:
    if not conversation_id:
        conversation_id = str(uuid.uuid4())

    seed_knowledge_base()

    initial_state = AgentState({
        "query": query,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "intent": None,
        "intent_confidence": 0.0,
        "agents_used": [],
        "sources": [],
        "agent_responses": {},
        "final_response": "",
        "confidence": 0.0,
        "error": "",
        "_context": {},
    })

    result = graph.invoke(initial_state)

    return ChatResponse(
        conversation_id=conversation_id,
        query=query,
        response=result.get("final_response", "No response generated."),
        intent=result.get("intent", IntentType.IMPACT),
        agents_used=result.get("agents_used", []),
        confidence=result.get("confidence", 0.5),
        sources=list(set(result.get("sources", []))),
        explanations=result.get("_context", {}).get("explanations"),
    )
