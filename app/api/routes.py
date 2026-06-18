import json
import random
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from ..models import ChatRequest, ChatResponse
from ..workflow.graph import run_chat
from ..memory.short_term import short_term_memory
from ..rag.vector_store import search_knowledge
from ..knowledge.neo4j_client import Neo4jClient
from ..event_memory.event_store import event_store
from ..event_memory.event_schema import HistoricalEvent
from ..explain.shap_explainer import SHAPExplainer
from ..explain.attention_explainer import AttentionExplainer
from ..explain.graph_explainer import GraphExplainer
from .data import COUNTRIES, COUNTRIES_BY_CODE, TRADE_ROUTES, MILITARY_RELATIONS, PORTS, LIVE_EVENTS

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = await run_chat(
            query=request.query,
            conversation_id=request.conversation_id,
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    response = await run_chat(
        query=request.query,
        conversation_id=request.conversation_id,
    )

    async def generate():
        yield json.dumps({
            "conversation_id": response.conversation_id,
            "intent": response.intent.value,
            "agents_used": response.agents_used,
            "confidence": response.confidence,
        }) + "\n"
        for chunk in response.response.split(". "):
            yield json.dumps({"chunk": chunk + ". "}) + "\n"
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/history")
async def history(limit: int = 20):
    try:
        from ..knowledge.postgres import get_conversation_history
        convs = await get_conversation_history(limit)
        return [
            {
                "id": c.id,
                "query": c.query[:100],
                "intent": c.intent,
                "confidence": c.confidence,
                "created_at": c.created_at.isoformat(),
            }
            for c in convs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/{conversation_id}")
async def get_memory(conversation_id: str):
    history = short_term_memory.get_history(conversation_id)
    return {"conversation_id": conversation_id, "turns": len(history), "history": history}


@router.get("/knowledge/search")
async def knowledge_search(q: str, limit: int = 5):
    results = search_knowledge(q, limit)
    return {"query": q, "results": results}


@router.get("/graph/{entity}")
async def graph_query(entity: str):
    client = Neo4jClient()
    if not client.available:
        return {"entity": entity, "error": "Neo4j not available", "relations": []}
    relations = client.get_relations(entity)
    return {"entity": entity, "relations": relations}


@router.get("/events/similar")
async def find_similar_events(
    q: str = Query(..., description="Query text describing the event"),
    entities: str = Query("", description="Comma-separated entity names"),
    sectors: str = Query("", description="Comma-separated sector names"),
    top_k: int = Query(5, description="Number of results"),
):
    entity_list = [e.strip() for e in entities.split(",") if e.strip()] if entities else None
    sector_list = [s.strip() for s in sectors.split(",") if s.strip()] if sectors else None

    results = event_store.find_similar(
        query_text=q,
        query_entities=entity_list,
        query_sectors=sector_list,
        top_k=top_k,
    )
    response = event_store.build_response(q, results, top_k=min(top_k, 3))

    return response


@router.get("/events")
async def list_events(
    skip: int = Query(0, description="Number of events to skip"),
    limit: int = Query(20, description="Max events to return"),
    type: str = Query(None, description="Filter by event type"),
    severity: str = Query(None, description="Filter by severity"),
):
    filtered = LIVE_EVENTS
    if type:
        filtered = [e for e in filtered if e["event_type"] == type]
    if severity:
        filtered = [e for e in filtered if e["severity"] == severity]
    items = filtered[skip:skip + limit]
    return {"total": len(filtered), "skip": skip, "limit": limit, "items": items}


@router.get("/events/{event_id}")
async def get_event(event_id: str):
    try:
        eid = int(event_id)
        for ev in LIVE_EVENTS:
            if ev["id"] == eid:
                return ev
    except ValueError:
        pass
    event = event_store.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.post("/events")
async def add_event(event: HistoricalEvent):
    event_store.add_event(event)
    return {"status": "added", "event_id": event.id, "name": event.name}


@router.post("/explain/shap")
async def explain_shap(query: str, prediction: str = ""):
    explainer = SHAPExplainer()
    result = explainer.explain(prediction=prediction, context={"query": query})
    shap = result.shap
    if shap:
        return shap.model_dump()
    return {"prediction": prediction, "contributions": []}


@router.post("/explain/attention")
async def explain_attention(query: str = "", entities: str = "", sectors: str = ""):
    explainer = AttentionExplainer()
    entity_list = [e.strip() for e in entities.split(",") if e.strip()] if entities else []
    sector_list = [s.strip() for s in sectors.split(",") if s.strip()] if sectors else []
    result = explainer.explain(context={"query": query, "entities": entity_list, "sectors": sector_list})
    attn = result.attention
    if attn:
        return attn.model_dump()
    return {"query": query, "top_events": [], "top_features": []}


@router.post("/explain/graph")
async def explain_graph(query: str = "", entities: str = ""):
    explainer = GraphExplainer()
    entity_list = [e.strip() for e in entities.split(",") if e.strip()] if entities else []
    result = explainer.explain(context={"query": query, "entities": entity_list})
    graph = result.graph
    if graph:
        return graph.model_dump()
    return {"start_entity": "", "path": []}


@router.get("/health")
async def health():
    return {"status": "ok", "service": "MarketAtlas Chat"}


@router.get("/countries")
async def list_countries():
    return COUNTRIES


@router.get("/countries/{code}")
async def get_country(code: str):
    c = COUNTRIES_BY_CODE.get(code.upper())
    if not c:
        raise HTTPException(status_code=404, detail="Country not found")
    return c


@router.get("/countries/{code}/relations/trade")
async def country_trade_routes(code: str):
    upper = code.upper()
    return [r for r in TRADE_ROUTES if r["from"] == upper or r["to"] == upper]


@router.get("/countries/{code}/relations/military")
async def country_military_relations(code: str):
    upper = code.upper()
    return [r for r in MILITARY_RELATIONS if r["countryA"] == upper or r["countryB"] == upper]


@router.get("/countries/{code}/ports")
async def country_ports(code: str):
    upper = code.upper()
    return [p for p in PORTS if p["countryCode"] == upper]


@router.get("/relations/trade")
async def all_trade_routes():
    return TRADE_ROUTES


@router.get("/relations/military")
async def all_military_relations():
    return MILITARY_RELATIONS


@router.get("/ports")
async def all_ports():
    return PORTS


_entity_market_data: dict[int, list[dict]] = {}


def _generate_market_data(entity_id: int) -> list[dict]:
    if entity_id in _entity_market_data:
        return _entity_market_data[entity_id]
    data = []
    base_price = random.uniform(50, 500)
    for i in range(60):
        date = (datetime.utcnow() - timedelta(days=59 - i)).strftime("%Y-%m-%d")
        change = random.uniform(-5, 5)
        o = round(base_price + change, 2)
        h = round(o + random.uniform(0, 3), 2)
        low_price = round(o - random.uniform(0, 3), 2)
        c = round(random.uniform(low_price, h), 2)
        v = random.randint(1000000, 50000000)
        data.append({
            "id": i + 1,
            "entity_id": entity_id,
            "open": o,
            "high": h,
            "low": low_price,
            "close": c,
            "volume": v,
            "price_date": date,
        })
        base_price = c
    _entity_market_data[entity_id] = data
    return data


@router.get("/market-prices/entity/{entity_id}/recent")
async def entity_market_prices(entity_id: int, days: int = 30):
    data = _generate_market_data(entity_id)
    items = data[-days:] if days < len(data) else data
    return {"items": items}


@router.get("/market-prices/entity/{entity_id}/latest")
async def entity_latest_price(entity_id: int):
    data = _generate_market_data(entity_id)
    if not data:
        raise HTTPException(status_code=404, detail="No data")
    return data[-1]


@router.post("/analyze")
async def analyze(body: dict):
    text = body.get("text", "")
    symbol = body.get("ticker") or _pick_symbol(text)

    q = text.lower()
    if any(w in q for w in ["energy", "oil", "gas", "xle"]):
        momentum = round(random.uniform(0.03, 0.12), 4)
        risk = round(random.uniform(0.5, 0.85), 4)
        action = "BUY"
        action_reason = "Energy sector strengthening amid geopolitical supply concerns."
    elif any(w in q for w in ["tech", "semiconductor", "xlk", "qqq"]):
        momentum = round(random.uniform(-0.03, 0.06), 4)
        risk = round(random.uniform(0.3, 0.6), 4)
        action = "HOLD" if risk < 0.5 else "SELL"
        action_reason = "Tech sector mixed with regulatory headwinds and valuation concerns."
    elif any(w in q for w in ["defense", "ita", "military"]):
        momentum = round(random.uniform(0.02, 0.10), 4)
        risk = round(random.uniform(0.4, 0.7), 4)
        action = "BUY"
        action_reason = "Defense spending outlook positive given geopolitical tensions."
    elif any(w in q for w in ["safe", "haven", "gold", "gld"]):
        momentum = round(random.uniform(0.01, 0.05), 4)
        risk = round(random.uniform(0.2, 0.4), 4)
        action = "BUY"
        action_reason = "Safe-haven demand increasing amid global uncertainty."
    else:
        momentum = round(random.uniform(-0.05, 0.08), 4)
        risk = round(random.uniform(0.3, 0.7), 4)
        action = "BUY" if momentum > 0 else "SELL"
        action_reason = "Mixed signals based on current market conditions."

    return {
        "snapshot": {
            "symbol": symbol,
            "momentum": momentum,
            "volatility": round(random.uniform(0.015, 0.05), 4),
            "volume_status": random.choice(["surge", "normal", "thin"]),
        },
        "impact": {
            "composite_risk": risk,
            "local_severity": round(random.uniform(0.2, 0.8), 4),
            "entity_count": random.randint(3, 10),
            "relations": [
                {"source": "Russia", "target": "Oil", "label": "sanction"},
                {"source": "China", "target": "Tech", "label": "restriction"},
            ],
        },
        "recommendation": {
            "action": action,
            "reason": action_reason,
            "confidence": round(random.uniform(0.6, 0.95), 4),
        },
    }


def _pick_symbol(text: str) -> str:
    text_lower = text.lower()
    if "energy" in text_lower or "oil" in text_lower or "gas" in text_lower:
        return "XLE"
    if "tech" in text_lower or "semiconductor" in text_lower:
        return "XLK"
    if "defense" in text_lower or "military" in text_lower:
        return "ITA"
    if "gold" in text_lower or "safe" in text_lower:
        return "GLD"
    if "financial" in text_lower or "bank" in text_lower:
        return "XLF"
    country_map = {
        "US": "SPY", "JP": "EWJ", "CN": "FXI", "GB": "EWU", "DE": "EWG",
        "IN": "INDA", "BR": "EWZ", "KR": "EWY", "TW": "EWT", "SG": "EWS",
        "AU": "EWA", "CA": "EWC", "MX": "EWW", "ZA": "EZA", "RU": "RSX",
    }
    for name, ticker in country_map.items():
        if name.lower() in text_lower:
            return ticker
    return random.choice(["SPY", "QQQ", "EEM", "XLE", "XLK", "GLD"])
