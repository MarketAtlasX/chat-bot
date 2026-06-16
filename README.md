# MarketAtlas Chat Bot

**AI-Powered Geopolitical Trading Intelligence Backend**

MarketAtlas Chat is a multi-agent conversational AI backend that transforms geopolitical events into actionable trading intelligence. It serves as the API layer for the MarketAtlas Frontend, providing REST endpoints, WebSocket real-time data, and a LangGraph-powered agent workflow.

```
User Query
    ↓
Intent Router (8 intents)
    ↓
LangGraph Agent Workflow
    ↓
Structured Response + Explanations
```

---

## Quick Start

```powershell
.\run.bat
```

Or step-by-step:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Opens at `http://localhost:8000`. All model caches stored in `.cache/` on D drive.

---

## Architecture

### Agent Workflow (LangGraph)

```
route_intent
    │
    ▼
decide_agents (conditional routing)
    │
    ├── execute_news
    ├── execute_market
    ├── execute_impact
    ├── execute_graph
    ├── execute_forecast
    ├── execute_recommendation
    ├── execute_simulation
    ├── execute_report
    ├── execute_similarity ──► EventSimilarityAgent
    ├── execute_similarity_pipeline ──► NewsAgent → EventSimilarityAgent → ImpactAgent → ForecastAgent → ReportAgent
    ├── execute_debate ──► Multi-analyst debate pipeline
    └── execute_direct ──► runs all agents in agents_used
    │
    ▼
calculate_confidence
    │
    ▼
store_memory (short-term + long-term)
    │
    ▼
END ──► ChatResponse
```

### Intent Routing

| Intent | Trigger Keywords | Agents Executed |
|--------|-----------------|-----------------|
| NEWS | news, latest, update, headline, sanctions | NewsAgent |
| MARKET | price, stock, market, ETF, index, trading | MarketAgent, NewsAgent |
| IMPACT | impact, affect, consequence, why is, tension | ImpactAgent, NewsAgent, MarketAgent |
| RECOMMENDATION | buy, sell, invest, should I, recommend, portfolio | RecommendationAgent, ImpactAgent, GraphAgent |
| SIMULATION | simulate, what if, scenario, if happens, if occurs | SimulationAgent, ImpactAgent |
| GRAPH | relationship, connection, how is, linked to, connection between | GraphAgent, NewsAgent |
| REPORT | report, brief, analysis, intelligence report, deep dive | ReportAgent, ImpactAgent, MarketAgent, GraphAgent, NewsAgent |
| SIMILARITY | similar, historical parallels, analogous, resemble, past events like | EventSimilarityAgent, ImpactAgent |

---

## 25 REST Endpoints

All prefixed with `/api/v1`.

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a query, receive structured ChatResponse with intent, confidence, sources, explanations |
| `POST` | `/chat/stream` | Streaming response as NDJSON (chunks with metadata) |
| `WebSocket` | `/ws` | WebSocket chat with stream/non-stream modes, plus channel subscriptions |

### Events (Historical Event Similarity Engine)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/events` | List all historical events. Supports `?skip=&limit=&type=&severity=` |
| `GET` | `/events/{id}` | Get single event by ID |
| `POST` | `/events` | Add a new historical event dynamically |
| `GET` | `/events/similar` | Find similar events. Params: `?q=&entities=&sectors=&top_k=` |

### Countries & Geospatial Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/countries` | List all 53 countries with market, ticker, commodity data |
| `GET` | `/countries/{code}` | Get single country by 2-letter code |
| `GET` | `/countries/{code}/relations/trade` | Trade routes for a country |
| `GET` | `/countries/{code}/relations/military` | Military relations (alliances, rivalries, conflicts) |
| `GET` | `/countries/{code}/ports` | Port locations for a country |
| `GET` | `/relations/trade` | All 40 global trade routes |
| `GET` | `/relations/military` | All 23 military relations |
| `GET` | `/ports` | All 69 port locations |

### Market Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/market-prices/entity/{id}/recent?days=` | Recent price history (simulated) |
| `GET` | `/market-prices/entity/{id}/latest` | Latest price point |
| `POST` | `/analyze` | Context-aware analysis returning snapshot, impact, recommendation for SignalDashboard |

### Explainability

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/explain/shap` | SHAP-style feature attribution for a prediction |
| `POST` | `/explain/attention` | Attention weights over events and features |
| `POST` | `/explain/graph` | Graph reasoning path tracing |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/history` | Recent conversation history |
| `GET` | `/memory/{id}` | Conversation turns for a session |
| `GET` | `/knowledge/search?q=&limit=` | RAG vector search |

---

## Agents (12 agents)

| Agent | File | Purpose |
|-------|------|---------|
| **Intent Router** | `intent_router.py` | Classifies queries into 8 intent types using keyword + LLM hybrid |
| **News Agent** | `news_agent.py` | Retrieves geopolitical news, extracts entities, queries Neo4j |
| **Market Agent** | `market_agent.py` | Extracts tickers, generates market snapshots + SHAP explanations |
| **Impact Agent** | `impact_agent.py` | Scores geopolitical risk, analyzes primary/secondary effects + SHAP + Graph explanations |
| **Graph Agent** | `graph_agent.py` | Queries Neo4j knowledge graph + Graph reasoning paths |
| **Forecast Agent** | `forecast_agent.py` | Probability-weighted forecasts with SHAP + Attention explanations |
| **Recommendation Agent** | `recommendation_agent.py` | BUY/HOLD/SELL guidance with conviction levels and picks |
| **Simulation Agent** | `simulation_agent.py` | What-if scenario analysis with structured JSON output |
| **Report Agent** | `report_agent.py` | Structured MarketAtlas Intelligence Reports |
| **Debate Agent** | `debate_agent.py` | Multi-analyst pipeline: Conflict → Energy → Market → Risk → Lead Intelligence Officer |
| **Event Similarity Agent** | `event_similarity_agent.py` | Finds historical analogs using multi-dimensional similarity |

---

## Event Similarity Engine

26 curated historical geopolitical events with market outcomes. Multi-dimensional similarity scoring:

```
similarity = 0.4 × text_similarity (BGE-M3 embedding)
           + 0.3 × entity_similarity (Jaccard entity overlap)
           + 0.2 × sector_similarity (Jaccard sector overlap)
           + 0.1 × market_similarity (outcome vector correlation)
```

Events covered: Yom Kippur War, Iranian Revolution, Gulf War, Asian Financial Crisis, 9/11, Iraq War, Global Financial Crisis, Arab Spring, Crimea Annexation, Iran Nuclear Deal, US-China Trade War, Strait of Hormuz Crisis, Saudi Oil Attack, Soleimani Assassination, COVID-19, OPEC+ Price War, Colonial Pipeline Cyberattack, US-Iran Shadow War, European Energy Crisis, Afghanistan Withdrawal, Russia-Ukraine War, Taiwan Strait Tensions, North Korea Missile Crisis, Sudan Civil War, Israel-Hamas War, Red Sea Shipping Crisis.

---

## Explainable Intelligence Layer

### SHAP-Style Feature Attribution

```python
Prediction: +3.1%

Contributions:
  Conflict Severity:     +3.1%
  Shipping Disruption:   +2.2%
  Inventory Decline:     +1.4%
  Negative Demand:       -0.8%
```

### Attention Weights

```python
Most Influential Events:
  1. Strait of Hormuz Crisis     — 22.74%
  2. Israel-Hamas War            — 20.70%
  3. Yom Kippur War              — 19.60%
```

### Graph Reasoning Paths

```python
Iran
  ↓ disrupts
Oil Supply
  ↓ prices
Energy Sector
  ↓ drives
XLE ETF
```

---

## MockLLM Fallback

When no Ollama server is available, `MockLLM` activates and produces comprehensive responses for ALL query types:

| Query Type | Response Quality |
|-----------|-----------------|
| **SIMILARITY** | Formatted historical event comparisons with similarity scores, entity/sector breakdown, and aggregated outcomes |
| **GRAPH** | Entity relationship trees with causal path tracing and cascading effects |
| **RECOMMENDATION** | BUY/HOLD/SELL with sector-specific picks, conviction level, allocation suggestions, and risk factors |
| **ENERGY** | Supply/driver/price-outlook analysis with sub-sector breakdown |
| **DEFENSE** | Sub-sector analysis with budget trends and order book context |
| **SANCTIONS** | Multi-sector impact analysis with trade flow implications |
| **CONFLICT** | Scenario-based risk assessment with base/escalation/de-escalation probabilities |
| **MARKET** | Sector rotation analysis, technical levels, VIX context |
| **TECH** | AI cycle, semiconductor supply chain, cybersecurity themes |
| **SAFE-HAVEN** | Gold, bonds, currency analysis with central bank context |
| **REPORT** | Proper JSON matching the IntelligenceReport Pydantic model |
| **GENERIC** | Entity + sector extraction with structured assessment |
| **JSON extraction** | Entities, sectors, tickers return valid JSON arrays |

---

## WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws`

### Client → Server

```json
// Subscribe to a channel (receives periodic updates)
{ "type": "subscribe", "channel": "signals" }
{ "type": "subscribe", "channel": "events" }

// Unsubscribe
{ "type": "unsubscribe", "channel": "signals" }

// Ping
{ "type": "ping" }

// Chat query (non-streaming)
{ "query": "Why is oil rising?", "conversation_id": "uuid" }

// Chat query (streaming)
{ "query": "Why is oil rising?", "stream": true, "conversation_id": "uuid" }
```

### Server → Client

```json
{ "type": "connected", "client_id": "uuid" }
{ "type": "subscribed", "channel": "signals" }
{ "type": "pong" }

// Periodic signal update (every 10s when subscribed)
{
  "type": "signal",
  "channel": "signals",
  "data": {
    "snapshot": { "symbol": "XLE", "momentum": 0.05, "volatility": 0.03, "volume_status": "surge" },
    "impact": { "composite_risk": 0.7, "local_severity": 0.4, "entity_count": 5, "relations": [...] },
    "recommendation": { "action": "BUY", "reason": "...", "confidence": 0.82 }
  },
  "timestamp": "2026-06-16T..."
}

// Chat response
{ "type": "response", "conversation_id": "uuid", "response": "...", "intent": "IMPACT", "confidence": 0.85 }
```

---

## Example Chat Queries

### Historical Parallels

```
Q: What historical events are similar to Iran-Israel tensions?
A: ## Historical Event Similarity Analysis
   1. Iran Nuclear Deal / JCPOA (2015) — Similarity: 82%
   2. Strait of Hormuz Crisis (2019) — Similarity: 79%
   3. Iran-Israel Shadow War (2021-2023) — Similarity: 76%
   Energy +12%, Defense +7%, Tech -3%
```

### Relationship Graph

```
Q: How is Russia connected to Europe energy?
A: Russia
     ├── supplies → Natural Gas (40% of EU imports)
     ├── supplies → Oil (25% of EU imports)
     ├── pipeline → Nord Stream
     └── rivalry → NATO
   Key Insight: Russia supplies ~40% of EU natural gas...
```

### Investment Recommendation

```
Q: Should I buy energy stocks?
A: Action: BUY | Sector: Energy | Conviction: High
   Top Picks: XLE, CVX, XOM
   Risks: Global recession, demand destruction, OPEC+ discord
```

---

## Project Structure

```
chat-bot/
├── main.py                          # FastAPI entry point
├── run.bat                          # One-click launcher
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── pytest.ini                       # Test config (19 tests)
├── app/
│   ├── config.py                    # Pydantic settings from env
│   ├── models.py                    # All Pydantic models
│   ├── agents/
│   │   ├── intent_router.py         # Query classifier
│   │   ├── news_agent.py            # News retrieval
│   │   ├── market_agent.py          # Market analysis + SHAP
│   │   ├── impact_agent.py          # Risk scoring + SHAP + Graph
│   │   ├── graph_agent.py           # Knowledge graph + Graph paths
│   │   ├── forecast_agent.py        # Forecasting + SHAP + Attention
│   │   ├── recommendation_agent.py  # Trade recommendations
│   │   ├── simulation_agent.py      # What-if scenarios
│   │   ├── report_agent.py          # Intelligence reports
│   │   ├── debate_agent.py          # Multi-analyst debate
│   │   └── event_similarity_agent.py # Similarity engine
│   ├── api/
│   │   ├── routes.py                # 25 REST endpoints
│   │   ├── websocket.py             # WebSocket handler
│   │   └── data.py                  # 53 countries, 40 routes, 69 ports, 10 events
│   ├── event_memory/
│   │   ├── event_schema.py          # HistoricalEvent Pydantic models
│   │   ├── event_data.py            # 26 curated historical events
│   │   └── event_store.py           # Similarity engine + Qdrant sync
│   ├── explain/
│   │   ├── base.py                  # BaseExplainer interface
│   │   ├── shap_explainer.py        # SHAP-style attribution
│   │   ├── attention_explainer.py   # Attention weights
│   │   └── graph_explainer.py       # Graph reasoning paths
│   ├── knowledge/
│   │   ├── postgres.py              # SQLAlchemy async models
│   │   └── neo4j_client.py          # Neo4j graph client
│   ├── llm/
│   │   ├── base.py                  # LLMInterface ABC
│   │   └── ollama.py                # OllamaLLM + MockLLM fallback
│   ├── memory/
│   │   ├── short_term.py            # Conversation buffer
│   │   └── long_term.py             # File-based persistence
│   ├── rag/
│   │   ├── embeddings.py            # BGE-M3 embeddings
│   │   ├── vector_store.py          # Qdrant client
│   │   └── retriever.py             # RAG pipeline
│   ├── utils/
│   │   ├── metrics.py               # Volatility, momentum, Sharpe
│   │   └── constants.py             # Sector→ETF map, keywords
│   └── workflow/
│       └── graph.py                 # LangGraph StateGraph
└── tests/
    └── test_chatbot.py              # 19 tests
```

---

## Testing

```powershell
python -m pytest tests/test_chatbot.py -v
```

All 19 tests cover: Intent routing (8), Similarity engine (3), NewsAgent (1), ImpactAgent (1), MarketAgent (1), GraphAgent (1), SimulationAgent (1), RecommendationAgent (1), ReportAgent (1), Knowledge base (1).

---

## Dependencies

- `fastapi`, `uvicorn` — API server
- `langgraph` — Agent workflow orchestration
- `sentence-transformers` — BGE-M3 embeddings
- `qdrant-client` — Vector search (optional)
- `neo4j` — Graph database (optional)
- `httpx` — Ollama HTTP client
- `pydantic` — Data validation
- `numpy` — Numerical operations

All external services (Ollama, Qdrant, Neo4j, PostgreSQL) are optional. The system works fully offline with MockLLM fallback.
