# MarketAtlas Chat

**AI-Powered Geopolitical Trading Intelligence Chatbot** — the backend intelligence engine for the MarketAtlas platform.

MarketAtlas Chat is a multi-agent conversational AI system that transforms geopolitical events into actionable trading insights. It uses a LangGraph-powered workflow with specialized agents, RAG retrieval, knowledge graphs, and structured report generation.

---

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Intent Router                                        │
│ (NEWS | MARKET | IMPACT | RECOMMENDATION |           │
│  SIMULATION | GRAPH | REPORT)                        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│ LangGraph Workflow                                    │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ News     │  │ Market   │  │ Impact           │   │
│  │ Agent    │  │ Agent    │  │ Agent            │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Graph    │  │ Forecast │  │ Recommendation   │   │
│  │ Agent    │  │ Agent    │  │ Agent            │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │Simulation│  │ Report   │  │ Debate           │   │
│  │ Agent    │  │ Agent    │  │ Agent            │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
           Structured Response + Report
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (Qwen / Mistral / Llama) + MockLLM fallback |
| Orchestration | LangGraph (StateGraph with conditional routing) |
| Vector Store | Qdrant + BGE-M3 embeddings |
| Graph DB | Neo4j (optional, best-effort) |
| SQL DB | PostgreSQL (async with SQLAlchemy) |
| Memory | Short-term (in-memory deque) + Long-term (JSON files) |
| API | FastAPI + WebSocket |
| Frontend Proxy | Vite (rewrites `/api` → `/api/v1`) |

---

## Agents

| Agent | File | Purpose |
|-------|------|---------|
| **Intent Router** | `app/agents/intent_router.py` | Classifies queries into 7 intent types using keyword + LLM hybrid approach |
| **News Agent** | `app/agents/news_agent.py` | Retrieves geopolitical news, extracts entities, queries Neo4j for context |
| **Market Agent** | `app/agents/market_agent.py` | Extracts tickers, generates market snapshots (momentum, volatility, volume) |
| **Impact Agent** | `app/agents/impact_agent.py` | Scores geopolitical risk (0-1), analyzes primary/secondary effects |
| **Graph Agent** | `app/agents/graph_agent.py` | Queries Neo4j knowledge graph for entity relationships (depth 2) |
| **Forecast Agent** | `app/agents/forecast_agent.py` | Generates probability-weighted forecasts with alternative scenarios |
| **Recommendation Agent** | `app/agents/recommendation_agent.py` | Produces BUY/HOLD/SELL guidance with conviction levels |
| **Simulation Agent** | `app/agents/simulation_agent.py` | Runs what-if scenario analysis with structured JSON output |
| **Report Agent** | `app/agents/report_agent.py` | Generates MarketAtlas Intelligence Reports (structured format) |
| **Debate Agent** | `app/agents/debate_agent.py` | Multi-analyst synthesis (Conflict → Energy → Market → Risk → Final) |

### Analyst Debate System

Instead of single-LLM responses, the Debate Agent runs a multi-perspective pipeline:

```
User Query
    │
    ▼
Conflict Analyst ──► Energy Analyst ──► Market Strategist ──► Risk Analyst
    │                    │                    │                    │
    └────────────────────┴────────────────────┴────────────────────┘
                                    │
                                    ▼
                          Lead Intelligence Officer
                           (Final Synthesis)
```

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### Chat

```bash
# Send a message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is oil rising today?"}'

# Response
{
  "conversation_id": "uuid",
  "query": "Why is oil rising today?",
  "response": "...",
  "intent": "IMPACT",
  "agents_used": ["ImpactAgent", "NewsAgent", "MarketAgent"],
  "confidence": 0.82,
  "sources": ["MarketAtlas Intelligence", "Reuters"]
}
```

### Streaming

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze oil markets"}'
# Returns NDJSON: {"chunk": "..."}\n{"chunk": "..."}\n{"done": true}
```

### WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/ws')

// Send
ws.send(JSON.stringify({ query: 'Why is oil rising?', stream: true }))

// Receive
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data)
  if (msg.type === 'chunk') console.log(msg.text)
  if (msg.type === 'stream_end') console.log('Done')
}
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Backend availability |
| `/api/v1/history?limit=20` | GET | Past conversations |
| `/api/v1/memory/{id}` | GET | Short-term conversation turns |
| `/api/v1/knowledge/search?q=&limit=5` | GET | RAG vector search |
| `/api/v1/graph/{entity}` | GET | Neo4j knowledge graph query |

---

## Example Queries

| Query | Intent | Agents Used |
|-------|--------|-------------|
| "Why is oil rising today?" | IMPACT | ImpactAgent, NewsAgent, MarketAgent |
| "What stocks benefit from a Taiwan blockade?" | MARKET | MarketAgent, NewsAgent |
| "Simulate Russia reducing gas exports by 30%" | SIMULATION | SimulationAgent, ImpactAgent |
| "Show latest sanctions" | NEWS | NewsAgent |
| "Should I buy energy stocks?" | RECOMMENDATION | RecommendationAgent, ImpactAgent, GraphAgent |
| "How is Russia connected to Europe energy?" | GRAPH | GraphAgent, NewsAgent |
| "Generate an intelligence report" | REPORT | ReportAgent, ImpactAgent, MarketAgent, GraphAgent, NewsAgent |

---

## RAG Pipeline

```
Query ──► BGE-M3 Embeddings ──► Qdrant Vector Search ──► Context Augmentation ──► LLM
```

- **Embeddings:** BGE-M3 model (1024-dim) via `sentence-transformers`, falls back to random vectors
- **Vector Store:** Qdrant with cosine similarity (graceful fallback when offline)
- **Seeded Knowledge:** 6 initial documents covering oil, sanctions, Taiwan blockade, etc.

---

## Memory System

- **Short-term:** In-memory deque (last 20 turns), accessible per conversation
- **Long-term:** JSON file per user ID stored in `memory_store/`
- **Persistence:** PostgreSQL `conversations` table for permanent history

---

## Project Structure

```
chat-bot/
├── main.py                       # FastAPI server entry point
├── run.bat                       # One-click launcher with auto venv setup
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic settings
│   ├── models.py                 # ChatRequest/Response, IntelligenceReport, etc.
│   ├── llm/
│   │   ├── base.py               # LLM interface
│   │   └── ollama.py             # Ollama + MockLLM fallback
│   ├── agents/
│   │   ├── intent_router.py      # Query classifier (7 intent types)
│   │   ├── news_agent.py         # Geopolitical news retrieval
│   │   ├── market_agent.py       # Market data analysis
│   │   ├── impact_agent.py       # Risk scoring
│   │   ├── graph_agent.py        # Knowledge graph queries
│   │   ├── forecast_agent.py     # Scenario forecasting
│   │   ├── recommendation_agent.py  # Trade recommendations
│   │   ├── simulation_agent.py   # What-if simulations
│   │   ├── report_agent.py       # Intelligence reports
│   │   └── debate_agent.py       # Multi-analyst debate
│   ├── memory/
│   │   ├── short_term.py         # Conversation history (20 turns)
│   │   └── long_term.py          # File-based persistent memory
│   ├── rag/
│   │   ├── embeddings.py         # BGE-M3 embeddings
│   │   ├── vector_store.py       # Qdrant client
│   │   └── retriever.py          # RAG pipeline + knowledge seeding
│   ├── knowledge/
│   │   ├── postgres.py           # SQLAlchemy async models
│   │   └── neo4j_client.py       # Neo4j graph operations
│   ├── workflow/
│   │   └── graph.py              # LangGraph workflow definition
│   └── api/
│       ├── routes.py             # REST endpoints
│       └── websocket.py          # WebSocket handler
└── tests/
    └── test_chatbot.py           # 14 tests covering all agents
```

---

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_chatbot.py -v
```

All 14 tests pass, covering:
- Intent routing for all 7 intent types
- Each agent's process method
- Knowledge base seeding
- Full workflow execution

---

## Offline / Mock Mode

When Ollama is unavailable, the `MockLLM` fallback activates automatically:
- Health check against Ollama API (2s timeout)
- Falls back to keyword-based responses with realistic market analysis text
- Supports structured JSON output for simulation and report agents
- All agents continue working without any external dependencies

---

## Frontend Integration

The chat-bot serves as the backend for the MarketAtlas frontend (at `github.com/MarketAtlasX/frontend`). The Vite dev server proxies `/api` → `http://localhost:8000/api/v1`.

See the [frontend README](https://github.com/MarketAtlasX/frontend) for the complete dashboard UI with 3D globe, country maps, signal dashboard, and integrated ChatBot.
