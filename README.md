# MarketAtlas Chat

**AI-Powered Geopolitical Trading Intelligence Chatbot**

MarketAtlas Chat is a multi-agent conversational AI system that transforms geopolitical events into actionable trading intelligence. It answers questions by retrieving live events, finding historical analogs, analyzing graph impact, forecasting markets, and generating explainable reports — all with transparent reasoning.

```
Question
    ↓
Retrieve Live Events
    ↓
Retrieve Historical Analogs
    ↓
Analyze Graph Impact
    ↓
Forecast Markets
    ↓
Generate Explainable Report
    ↓
Answer + Explanations
```

---

## Quick Start

```powershell
run.bat
```

Or step by step:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000

All ML model caches are stored in the project's `.cache/` directory (D drive, not C drive). Run `scripts/download_model.py` once to enable BGE-M3 semantic embeddings.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Intent Router  (8 intents: NEWS | MARKET | IMPACT | RECOMMEND   │
│  SIMULATION | GRAPH | REPORT | SIMILARITY)                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ LangGraph Workflow                                               │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ News     │  │ Market   │  │ Impact     │  │ Graph        │  │
│  │ Agent    │  │ Agent    │  │ Agent      │  │ Agent        │  │
│  └──────────┘  └──────────┘  └────────────┘  └──────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │Forecast  │  │Recommend │  │ Simulation  │  │ Report       │  │
│  │ Agent    │  │ Agent    │  │ Agent       │  │ Agent        │  │
│  └──────────┘  └──────────┘  └────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Event        │  │ Debate       │  │ Explainable          │  │
│  │ Similarity   │  │ Agent        │  │ Intelligence Layer   │  │
│  │ Agent        │  │              │  │ (SHAP/Attention/Graph)│  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
            Structured Response + Explanations
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Ollama (Qwen / Mistral / Llama) + MockLLM fallback |
| **Orchestration** | LangGraph (StateGraph with conditional routing) |
| **Vector Store** | Qdrant + BGE-M3 embeddings (1024-dim) |
| **Graph DB** | Neo4j (optional, best-effort fallback) |
| **SQL DB** | PostgreSQL (async with SQLAlchemy, optional) |
| **Memory** | Short-term (in-memory deque) + Long-term (JSON files) |
| **API** | FastAPI + WebSocket |
| **Explainability** | SHAP-style feature attribution, Attention weights, Graph reasoning paths |

---

## Event Similarity Engine

MarketAtlas can answer questions like "What historical events are similar to the current Iran-Israel tensions?" with evidence-based answers.

### Pipeline

```
New Event
    ↓
BGE-M3 Embedding (or keyword fallback)
    ↓
Multi-Dimensional Vector Search
    ↓
Top K Similar Events with Similarity Scores
```

### Multi-Dimensional Similarity

Similarity combines four dimensions with learned weights:

```
similarity = 0.4 × text_similarity
           + 0.3 × entity_similarity
           + 0.2 × sector_similarity
           + 0.1 × market_similarity
```

This enables cross-domain pattern matching. For example, Russia-Ukraine War and a Taiwan Blockade have low text similarity but high sector and supply chain overlap — resulting in a high combined score.

### Historical Outcome Retrieval

For each matched event, the system retrieves:

```
Event → Market Reaction → Volatility → Recovery Time
```

Then answers: *"Historically similar events produced a 12% rise in oil."*

### Curated Events (26 total)

| Era | Events |
|-----|--------|
| **1970s** | Yom Kippur War, Iranian Revolution |
| **1990s** | Gulf War, Asian Financial Crisis |
| **2000s** | 9/11, Iraq War, Global Financial Crisis |
| **2010s** | Arab Spring, Crimea Annexation, US-China Trade War, Strait of Hormuz Crisis, North Korea Missile Crisis, Iran Nuclear Deal |
| **2020s** | COVID-19, Russia-Ukraine War, Taiwan Strait Tensions, US-Iran Shadow War, Soleimani Assassination, Israel-Hamas War, Red Sea Shipping Crisis, Sudan Conflict, European Energy Crisis, Colonial Pipeline Cyberattack, Afghanistan Withdrawal, Saudi Oil Attack, OPEC Price War |

### Example Response

```
## Query: What historical events are similar to the current Iran-Israel tensions?

### Similar Historical Events

1. Iranian Revolution (1979)
   Similarity: 87%

2. Strait of Hormuz Crisis (2019)
   Similarity: 82%

3. Israel-Hamas War (2023)
   Similarity: 78%

### Historical Outcomes

   Energy: +15.0%
   Defense: +10.0%
   Airlines: -6.0%

Confidence: 81%
```

---

## Explainable Intelligence Layer

Every prediction includes transparent explanations.

### SHAP-Style Feature Attribution

Instead of "BUY XLE", the system explains:

```
Prediction: +3.1%

SHAP Output:
   Conflict Severity:     +3.1%
   Shipping Disruption:   +2.2%
   Inventory Decline:     +1.4%
   Negative Demand:       -0.8%

Reasons:
   40%  Middle East Conflict
   25%  Oil Inventory Decline
   20%  Positive Momentum
   15%  Historical Analog Events

Confidence: 81%
```

### Attention Weights

For transformer-based event analysis, the system displays which inputs drove the prediction:

```
Most Important Events:
   1. Strait of Hormuz Crisis     — 22.74%
   2. OPEC Statement              — 20.70%
   3. Shipping Delays             — 17.67%

Key Features:
   • Geopolitical Risk: 56.3%
   • Energy Prices:     43.7%
```

### Graph Reasoning Paths

For relationship queries, the system traces causal chains through the knowledge graph:

```
Reasoning Path:

Iran
  ↓ disrupts
Oil Supply
  ↓ prices
Energy Sector
  ↓ drives
XLE ETF
```

This uses Neo4j when available, with an intelligent fallback that maps entities → sectors → tickers.

---

## Agents (12 agents)

### Core Agents

| Agent | File | Purpose |
|-------|------|---------|
| **Intent Router** | `intent_router.py` | Classifies queries into 8 intent types using keyword + LLM hybrid |
| **News Agent** | `news_agent.py` | Retrieves geopolitical news, extracts entities, queries Neo4j for graph context |
| **Market Agent** | `market_agent.py` | Extracts tickers, generates market snapshots (momentum, volatility, volume) + SHAP explanations |
| **Impact Agent** | `impact_agent.py` | Scores geopolitical risk (0-1), analyzes primary/secondary effects + SHAP + Graph explanations |
| **Graph Agent** | `graph_agent.py` | Queries Neo4j knowledge graph for entity relationships + Graph reasoning path |
| **Forecast Agent** | `forecast_agent.py` | Generates probability-weighted forecasts with alternative scenarios + SHAP + Attention explanations |
| **Recommendation Agent** | `recommendation_agent.py` | Produces BUY/HOLD/SELL guidance with conviction levels |
| **Simulation Agent** | `simulation_agent.py` | Runs what-if scenario analysis with structured JSON output |
| **Report Agent** | `report_agent.py` | Generates MarketAtlas Intelligence Reports (structured format + free-form fallback) |
| **Debate Agent** | `debate_agent.py` | Multi-analyst pipeline: Conflict → Energy → Market → Risk → Lead Intelligence Officer |
| **Event Similarity Agent** | `event_similarity_agent.py` | Finds historical analogs using multi-dimensional similarity + Attention + Graph explanations |

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

### Similarity Pipeline

For SIMILARITY queries, a dedicated 5-agent pipeline runs automatically:

```
Similarity Query
    │
    ▼
News Agent ──► EventSimilarityAgent ──► Impact Agent ──► Forecast Agent ──► Report Agent
    │                │                       │                 │                 │
    ▼                ▼                       ▼                 ▼                 ▼
Live Events    Historical Analogs     Impact Analysis      Forecast       Final Report
               + Attention Weights    + SHAP + Graph                        + Summary
               + Graph Paths
```

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### Chat

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What historical events are similar to Iran-Israel tensions?"}'

# Response includes explanations
{
  "conversation_id": "uuid",
  "query": "...",
  "response": "### Similar Historical Events\n\n1. ...",
  "intent": "SIMILARITY",
  "agents_used": ["NewsAgent", "EventSimilarityAgent", "ImpactAgent", "ForecastAgent", "ReportAgent"],
  "confidence": 0.85,
  "sources": ["News API", "Event Memory Database", "Knowledge Base"],
  "explanations": {
    "EventSimilarityAgent": {
      "attention": { "top_events": [...], "top_features": [...] },
      "graph": { "path": [...], "path_summary": "..." }
    },
    "ImpactAgent": {
      "shap": { "contributions": [...], "predicted_change_pct": 1.7 },
      "graph": { "path": [...] }
    }
  }
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

ws.send(JSON.stringify({ query: 'What historical parallels exist?', stream: true }))

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data)
  if (msg.type === 'chunk') console.log(msg.text)
  if (msg.type === 'stream_end') console.log('Done')
}
```

### Event Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/events` | GET | List all historical events |
| `/api/v1/events/{id}` | GET | Get specific event details |
| `/api/v1/events` | POST | Add a new historical event dynamically |
| `/api/v1/events/similar?q=...&entities=...&sectors=...&top_k=5` | GET | Find similar historical events |

### Explainability Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/explain/shap` | POST | SHAP-style feature attribution |
| `/api/v1/explain/attention` | POST | Attention weight analysis |
| `/api/v1/explain/graph` | POST | Graph reasoning path tracing |

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
| "What historical events are similar to Iran-Israel tensions?" | SIMILARITY | NewsAgent, EventSimilarityAgent, ImpactAgent, ForecastAgent, ReportAgent |
| "Find past events like the Russia-Ukraine war" | SIMILARITY | EventSimilarityAgent, ImpactAgent |
| "Why is oil rising today?" | IMPACT | ImpactAgent, NewsAgent, MarketAgent |
| "What stocks benefit from a Taiwan blockade?" | MARKET | MarketAgent, NewsAgent |
| "Simulate Russia reducing gas exports by 30%" | SIMULATION | SimulationAgent, ImpactAgent |
| "Show latest sanctions" | NEWS | NewsAgent |
| "Should I buy energy stocks?" | RECOMMENDATION | RecommendationAgent, ImpactAgent, GraphAgent |
| "How is Russia connected to Europe energy?" | GRAPH | GraphAgent, NewsAgent |
| "Generate an intelligence report on Iran" | REPORT | ReportAgent, ImpactAgent, MarketAgent, GraphAgent, NewsAgent |

---

## Explainability Examples

### SHAP Feature Attribution

```python
from app.explain.shap_explainer import SHAPExplainer

explainer = SHAPExplainer()
result = explainer.explain(context={
    "query": "Iran-Israel tensions impacting oil markets",
    "entities": ["Iran", "Israel"],
    "sectors": ["Energy", "Defense"]
})

for c in result.shap.contributions:
    print(f"{c.feature}: {c.impact_pct:+.1f}% ({c.direction})")
# Output:
#   Conflict Severity: +1.7% (positive)
#   Energy Supply Risk: +2.9% (positive)
```

### Attention Weights

```python
from app.explain.attention_explainer import AttentionExplainer

explainer = AttentionExplainer()
result = explainer.explain(context={
    "query": "How will Strait of Hormuz tensions affect oil?",
    "entities": ["Iran", "Strait of Hormuz"],
    "sectors": ["Energy", "Shipping"]
})

for e in result.attention.top_events:
    print(f"{e.input_label}: {e.weight:.1%}")
# Output:
#   Strait of Hormuz Crisis: 22.74%
#   OPEC Statement: 20.70%
#   Shipping Delays: 17.67%
```

### Graph Reasoning Path

```python
from app.explain.graph_explainer import GraphExplainer

explainer = GraphExplainer()
result = explainer.explain(context={
    "entities": ["Iran"],
    "sectors": ["Energy"]
})

for step in result.graph.path:
    print(f"{step.source} --[{step.relation}]--> {step.target}")
# Output:
#   Iran --[disrupts]--> Energy
#   Energy --[prices]--> XLE
```

---

## RAG Pipeline

```
Query ──► BGE-M3 Embeddings ──► Qdrant Vector Search ──► Context Augmentation ──► LLM
```

- **Embeddings:** BGE-M3 model (1024-dim) via `sentence-transformers`, falls back to random vectors when model is not cached
- **Vector Store:** Qdrant with cosine similarity (graceful fallback when offline)
- **Seeded Knowledge:** 6 initial documents covering oil, sanctions, Taiwan blockade, Fed decisions, gold rally
- **Cache:** All model files stored in project `.cache/` on D drive (not C drive)

---

## Memory System

- **Short-term:** In-memory deque (last 20 turns), accessible per conversation
- **Long-term:** JSON file per user ID stored in `memory_store/`
- **Persistence:** PostgreSQL `conversations` table for permanent history (optional)

---

## Offline / Mock Mode

When Ollama is unavailable, the `MockLLM` fallback activates automatically:

- Falls back to keyword-based intent classification with realistic market analysis text
- Supports structured JSON output for simulation and report agents
- Entity extraction and sector detection work from keywords alone
- All 12 agents continue working without any external dependencies
- Embeddings fall back to random vectors when BGE-M3 model is not cached

---

## Project Structure

```
chat-bot/
├── main.py                          # FastAPI server entry point
├── run.bat                          # One-click launcher with D drive cache routing
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules (includes .cache/)
├── config.example.yaml              # Alternative YAML configuration
├── pytest.ini                       # Test configuration
├── scripts/
│   ├── download_model.py            # Download BGE-M3 to D drive cache
│   └── __init__.py
├── app/
│   ├── config.py                    # Pydantic settings from env vars
│   ├── models.py                    # All Pydantic models (requests, responses, explanations)
│   ├── agents/
│   │   ├── intent_router.py         # Query classifier (8 intent types)
│   │   ├── news_agent.py            # Geopolitical news retrieval
│   │   ├── market_agent.py          # Market data + SHAP explanations
│   │   ├── impact_agent.py          # Risk scoring + SHAP + Graph explanations
│   │   ├── graph_agent.py           # Knowledge graph + Graph reasoning paths
│   │   ├── forecast_agent.py        # Scenario forecasting + SHAP + Attention
│   │   ├── recommendation_agent.py  # Trade recommendations
│   │   ├── simulation_agent.py      # What-if simulations
│   │   ├── report_agent.py          # Intelligence reports
│   │   ├── debate_agent.py          # Multi-analyst debate
│   │   └── event_similarity_agent.py # Historical event similarity + explanations
│   ├── event_memory/
│   │   ├── event_schema.py          # HistoricalEvent, SimilarityResult models
│   │   ├── event_data.py            # 26 curated historical events with market outcomes
│   │   └── event_store.py           # Similarity engine + Qdrant sync + outcome aggregation
│   ├── explain/
│   │   ├── base.py                  # BaseExplainer interface with formatted output
│   │   ├── models.py                # Explanation Pydantic models
│   │   ├── shap_explainer.py        # SHAP-style feature attribution
│   │   ├── attention_explainer.py   # Attention weight analysis
│   │   └── graph_explainer.py       # Neo4j reasoning path tracing
│   ├── knowledge/
│   │   ├── postgres.py              # SQLAlchemy async models (optional)
│   │   └── neo4j_client.py          # Neo4j graph operations (optional)
│   ├── llm/
│   │   ├── base.py                  # LLM interface ABC
│   │   └── ollama.py                # Ollama + MockLLM fallback
│   ├── memory/
│   │   ├── short_term.py            # Conversation history (20 turns)
│   │   └── long_term.py             # File-based persistent memory
│   ├── rag/
│   │   ├── embeddings.py            # BGE-M3 with D drive cache
│   │   ├── vector_store.py          # Qdrant client
│   │   └── retriever.py             # RAG pipeline + knowledge seeding
│   ├── utils/
│   │   ├── metrics.py               # Volatility, momentum, Sharpe ratio
│   │   └── constants.py             # Sector→ETF mapping, keywords, event types
│   ├── workflow/
│   │   └── graph.py                 # LangGraph workflow definition
│   └── api/
│       ├── routes.py                # REST endpoints (chat, events, explain)
│       └── websocket.py             # WebSocket handler
├── tests/
│   └── test_chatbot.py              # 19 tests covering all agents and similarity engine
└── memory_store/                    # Long-term memory JSON files (runtime)
```

---

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests/test_chatbot.py -v
```

All 19 tests pass, covering:
- Intent routing for all 8 intent types (NEWS, MARKET, IMPACT, RECOMMENDATION, SIMULATION, GRAPH, REPORT, SIMILARITY)
- Event similarity engine (finding matches, formatting responses, aggregating outcomes)
- Each agent's process method (News, Impact, Market, Graph, Simulation, Recommendation, Report)
- Knowledge base seeding

---

## Development

### Adding New Historical Events

```python
from app.event_memory.event_store import event_store
from app.event_memory.event_schema import HistoricalEvent, MarketOutcome

event = HistoricalEvent(
    id="my-event-2026",
    name="My Geopolitical Event (2026)",
    description="Description of the event...",
    date="2026-06-01",
    event_type="conflict",
    entities=["CountryA", "CountryB"],
    sectors=["Energy", "Defense"],
    outcomes=[MarketOutcome(sector="Energy", impact_pct=10.0, volatility=45.0, recovery_days=90)],
    volatility=45.0,
    recovery_days=90,
    summary="Brief summary of the event and market impact."
)

event_store.add_event(event)  # Adds to in-memory store + Qdrant
```

Or via API:

```bash
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{"id": "my-event", "name": "...", ...}'
```

### Downloading BGE-M3

For high-quality semantic embeddings (vs keyword fallback):

```powershell
python scripts/download_model.py
```

This downloads ~2.2GB to `.cache/sentence_transformers/` on D drive.

---

## Environment Variables

| Variable | Default | Required |
|----------|---------|----------|
| `LLM_PROVIDER` | `ollama` | No |
| `LLM_MODEL` | `qwen2.5:7b` | No |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | No |
| `POSTGRES_DSN` | `postgresql+asyncpg://...` | No |
| `NEO4J_URI` | `bolt://localhost:7687` | No |
| `NEO4J_USER` | `neo4j` | No |
| `NEO4J_PASSWORD` | `test` | No |
| `QDRANT_URL` | `http://localhost:6333` | No |
| `QDRANT_COLLECTION` | `marketatlas_events` | No |
| `API_HOST` | `0.0.0.0` | No |
| `API_PORT` | `8000` | No |

All external services are optional — the system works fully with the MockLLM fallback.

---

## Frontend Integration

This backend serves the MarketAtlas frontend (at `github.com/MarketAtlasX/frontend`). The Vite dev server proxies `/api` → `http://localhost:8000/api/v1`. The dashboard includes a 3D globe, country maps, signal dashboard, and integrated ChatBot with explanation panels.
