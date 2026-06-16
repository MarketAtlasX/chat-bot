import json
import uuid
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
async def list_events():
    events = event_store.get_all_events()
    return [
        {
            "id": e.id,
            "name": e.name,
            "date": e.date,
            "event_type": e.event_type,
            "entities": e.entities,
            "sectors": e.sectors,
            "summary": e.summary,
        }
        for e in events
    ]


@router.get("/events/{event_id}")
async def get_event(event_id: str):
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
async def explain_attention(query: str, entities: str = "", sectors: str = ""):
    explainer = AttentionExplainer()
    entity_list = [e.strip() for e in entities.split(",") if e.strip()] if entities else []
    sector_list = [s.strip() for s in sectors.split(",") if s.strip()] if sectors else []
    result = explainer.explain(context={"query": query, "entities": entity_list, "sectors": sector_list})
    attn = result.attention
    if attn:
        return attn.model_dump()
    return {"query": query, "top_events": [], "top_features": []}


@router.post("/explain/graph")
async def explain_graph(query: str, entities: str = ""):
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
