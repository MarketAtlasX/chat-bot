import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn

from app.api.routes import router
from app.api.websocket import handle_websocket
from app.config import settings

app = FastAPI(
    title="MarketAtlas Chat",
    description="AI-Powered Geopolitical Trading Intelligence Chatbot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return RedirectResponse(url="/api/v1/health")


@app.get("/health")
async def health_redirect():
    return {"status": "ok", "service": "MarketAtlas Chat"}


@app.get("/api/v1")
async def api_root():
    return {"service": "MarketAtlas Chat API", "version": "1.0.0", "endpoints": [
        "/chat", "/chat/stream", "/health", "/history", "/memory/{id}",
        "/knowledge/search", "/graph/{entity}",
        "/events", "/events/{event_id}", "/events/similar",
        "/explain/shap", "/explain/attention", "/explain/graph",
        "/countries", "/countries/{code}",
        "/countries/{code}/relations/trade",
        "/countries/{code}/relations/military",
        "/countries/{code}/ports",
        "/relations/trade", "/relations/military", "/ports",
        "/market-prices/entity/{entity_id}/recent",
        "/market-prices/entity/{entity_id}/latest",
        "/analyze",
    ]}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await handle_websocket(websocket)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
