import json
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from ..models import ChatRequest, ChatResponse
from ..workflow.graph import run_chat
from ..memory.short_term import short_term_memory
from ..rag.vector_store import search_knowledge
from ..knowledge.neo4j_client import Neo4jClient

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


@router.get("/health")
async def health():
    return {"status": "ok", "service": "MarketAtlas Chat"}
