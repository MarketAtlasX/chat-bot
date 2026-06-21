# MarketAtlas Chat Bot

**AI-Powered Geopolitical Trading Intelligence Backend**

MarketAtlas Chat is a multi-agent conversational AI backend that transforms geopolitical events into actionable trading intelligence. It serves as the API layer for the MarketAtlas Frontend, providing REST endpoints, WebSocket real-time data, a LangGraph-powered agent workflow, and a comprehensive **GeoRAG pipeline system**.

```
User Query
    ↓
Intent Router (8 intents)
    ↓
LangGraph Agent Workflow
    ↓
GeoRAG Multi-Source Retrieval
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

### GeoRAG: Why Normal RAG is Not Enough

Most RAG systems follow a simple pipeline:

```
Question → Vector Search → Top 5 Documents → LLM → Answer
```

MarketAtlas requires a fundamentally different approach — **GeoRAG** — because geopolitical analysis is inherently multi-dimensional:

```
Question
    ↓
Intent Detection (7 intent types)
    ↓
Multi-Source Parallel Retrieval
    ├── News Articles (Qdrant vector search)
    ├── Historical Event Analogs (multi-factor similarity)
    ├── Knowledge Graph Paths (entity relationships)
    └── Market Reactions (historical market responses)
    ↓
Cross-Encoder Reranking (BGE-Reranker)
    ↓
Context Assembly
    ↓
LLM Reasoning
    ↓
Answer
```

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

## GeoRAG Pipeline System (`rag/`)

The GeoRAG system is a modular, extensible RAG framework purpose-built for geopolitical intelligence. It lives at `rag/` and consists of **10 modules across 20+ source files**.

### Module Overview

```
rag/
├── __init__.py                  # Top-level exports + seed_knowledge_base()
├── ingestion/                   # Document ingestion pipeline
│   ├── __init__.py
│   └── news_ingestor.py         # Chunk → Embed → Store flow
├── chunking/                    # Text splitting strategies
│   ├── __init__.py
│   └── text_chunker.py          # 5 chunking strategies
├── embeddings/                  # Vector embedding generation
│   ├── __init__.py
│   └── bge_m3.py                # BGE-M3 with singleton pattern
├── vectorstore/                 # Vector database abstraction
│   ├── __init__.py
│   └── qdrant_store.py          # Qdrant client with graceful degradation
├── retrievers/                  # Specialized retrievers
│   ├── __init__.py
│   ├── base.py                  # BaseRetriever ABC + type definitions
│   ├── news_retriever.py        # News article vector search
│   ├── market_retriever.py      # 10 curated market reactions
│   ├── historical_retriever.py  # 26 curated geopolitical events
│   ├── graph_retriever.py       # 13-entity knowledge graph
│   └── multi_retriever.py       # Parallel multi-source orchestrator
├── rerankers/
│   ├── __init__.py
│   └── bge_reranker.py          # BGE cross-encoder + keyword fallback
├── historical_memory/           # Event similarity engine
│   ├── __init__.py
│   ├── event_embeddings.py      # Pre-computed event embeddings
│   ├── event_similarity.py      # Multi-factor similarity scoring
│   └── analog_retriever.py      # Historical analog finder
├── graph_retrieval/             # Knowledge graph engine
│   ├── __init__.py
│   ├── neo4j_client.py          # Neo4j database client
│   ├── graph_query.py           # Entity relationship queries
│   └── graph_paths.py           # Multi-hop path extraction
├── geo_rag/                     # Main orchestration layer
│   ├── __init__.py
│   ├── intent_classifier.py     # 7-intent classifier + entity/sector/region extraction
│   ├── context_builder.py       # Multi-source context assembly
│   └── main_pipeline.py         # GeoRAGPipeline orchestrator
└── pipelines/                   # Production-ready pipelines
    ├── __init__.py
    ├── news_rag.py              # News RAG Pipeline
    ├── historical_similarity.py # Historical Event Similarity Pipeline
    ├── graph_rag.py             # Graph RAG Pipeline
    ├── market_rag.py            # Market RAG Pipeline
    ├── explainability_rag.py    # Explainability RAG Pipeline
    └── multi_geo_rag.py         # Multi-Retriever GeoRAG (main production pipeline)
```

### Pipeline 1: News RAG Pipeline

**Purpose:** Retrieve relevant geopolitical news articles via vector search.

```
News → Chunking → Embeddings → Qdrant → Vector Search → Results
```

```python
from rag.pipelines import NewsRAGPipeline

pipeline = NewsRAGPipeline()
result = await pipeline.query("What are the latest Iran sanctions?")
# result.context contains formatted news context
```

### Pipeline 2: Historical Event Similarity RAG

**Purpose:** Find historical events that resemble a current situation — a key innovation.

```
Current Event → Embedding → Multi-Factor Similarity → Historical Events → Ranked Analogs
```

Similarity scoring breakdown:
- **40%** — Text similarity (BGE-M3 embedding cosine distance)
- **30%** — Entity similarity (Jaccard overlap of named entities)
- **20%** — Sector similarity (energy, technology, defense, etc.)
- **10%** — Region similarity (Middle East, Europe, Asia Pacific, etc.)

```python
from rag.pipelines import HistoricalSimilarityPipeline

pipeline = HistoricalSimilarityPipeline()
result = pipeline.find_analogs("What historical events resemble Taiwan tensions?")
# result.analogs[0].event_name → "Taiwan Straits Crisis (2022)"
```

26 curated events across 7 decades: Oil crises, wars, sanctions, trade wars, pandemics, financial crises, and supply chain disruptions.

### Pipeline 3: Graph RAG

**Purpose:** Retrieve entity relationships and causal paths from a knowledge graph.

```
Question → Entity Extraction → Graph Traversal → Path Extraction → Structured Context
```

```python
from rag.pipelines import GraphRAGPipeline

pipeline = GraphRAGPipeline()
result = await pipeline.query("How does Iran affect European energy?")
# result.paths[0].path_string → "Iran[supplies oil to]→Europe"
```

The in-memory knowledge graph covers **13 entities** (countries, regions, organizations, chokepoints, industries) with **60+ relationships** across 5 sectors. Neo4j integration is available when the graph database is running.

### Pipeline 4: Market RAG

**Purpose:** Retrieve historical market reactions to geopolitical events.

```
Query → Event Matching → Market Database → Reaction Retrieval
```

```python
from rag.pipelines import MarketRAGPipeline

pipeline = MarketRAGPipeline()
result = await pipeline.query("What happened to oil during similar crises?")
# result.reactions[0].event → "Iran sanctions 2018"
# result.reactions[0].reaction → "prices surged 25% over 3 months"
```

10 curated market reactions covering oil, gas, equities, semiconductors, shipping, and safe-haven assets.

### Pipeline 5: Explainability RAG

**Purpose:** Instead of just retrieving documents, retrieve prediction factors, graph paths, historical analogs, and reasoning chains — this is rare in RAG systems.

```
Prediction → SHAP-style Factor Retrieval → Historical Analogs → Graph Paths → Explanation
```

```python
from rag.pipelines import ExplainabilityRAGPipeline

pipeline = ExplainabilityRAGPipeline()
result = await pipeline.explain("BUY XLE", prediction_type="energy")
# result.explanation → "The prediction is driven by 5 key factors..."
# result.reasoning_factors → [...]
# result.historical_analogs → [...]
# result.graph_paths → [...]
```

Sector-specific explanation templates for: energy, technology, defense, finance, and general.

### Pipeline 6: Multi-Retriever GeoRAG (Main Production Pipeline)

**Purpose:** The primary pipeline that combines everything — intent classification, parallel multi-source retrieval, reranking, and context assembly.

```
Question → Intent Classification → Parallel Retrieval → Reranking → Context → LLM Prompt
```

```python
from rag.pipelines import MultiGeoRAGPipeline

pipeline = MultiGeoRAGPipeline()
result = await pipeline.query("Will Taiwan tensions affect Nvidia?")

# result.intent → "multi_source" (automatically detected)
# result.sources_used → ["historical", "graph", "market"]
# result.combined_context → multi-source context string
# result.prompt → ready-to-send LLM prompt with full context
```

The intent classifier automatically detects the question type and routes to the appropriate retrievers:

| Intent | Retrievers Activated |
|--------|---------------------|
| `news_analysis` | News |
| `historical_analog` | Historical + Analog |
| `market_impact` | Market + News |
| `graph_relationship` | Graph |
| `geopolitical_risk` | News + Historical + Graph |
| `general_query` | News + Historical |
| `multi_source` | News + Historical + Graph + Market |

---

## API Reference

### 25 REST Endpoints

All prefixed with `/api/v1`.

#### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a query, receive structured ChatResponse with intent, confidence, sources, explanations |
| `POST` | `/chat/stream` | Streaming response as NDJSON (chunks with metadata) |
| `WebSocket` | `/ws` | WebSocket chat with stream/non-stream modes, plus channel subscriptions |

#### Events (Historical Event Similarity Engine)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/events` | List all historical events. Supports `?skip=&limit=&type=&severity=` |
| `GET` | `/events/{id}` | Get single event by ID |
| `POST` | `/events` | Add a new historical event dynamically |
| `GET` | `/events/similar` | Find similar events. Params: `?q=&entities=&sectors=&top_k=` |

#### Countries & Geospatial Data

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

#### Market Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/market-prices/entity/{id}/recent?days=` | Recent price history (simulated) |
| `GET` | `/market-prices/entity/{id}/latest` | Latest price point |
| `POST` | `/analyze` | Context-aware analysis returning snapshot, impact, recommendation for SignalDashboard |

#### Explainability

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/explain/shap` | SHAP-style feature attribution for a prediction |
| `POST` | `/explain/attention` | Attention weights over events and features |
| `POST` | `/explain/graph` | Graph reasoning path tracing |

#### System

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

Events covered: Iran Nuclear Deal Collapse, Russia-Ukraine War, Taiwan Straits Crisis, Hormuz Strait Disruption, US-China Trade War, Gulf War, COVID-19 Pandemic, Suez Canal Blockage, Libyan Civil War, Korean Peninsula Tensions, Annexation of Crimea, OPEC Oil Crisis, Japan Tsunami & Fukushima, Fall of the Berlin Wall, Asian Financial Crisis, 9/11 Attacks, Hamas Attack on Israel, Brexit Referendum, South China Sea Tensions, Sri Lanka Economic Collapse, US Debt Ceiling Crisis, Cyprus Financial Crisis, Venezuela Collapse, China Evergrande Crisis, Global Chip Shortage, AUKUS Pact.

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

### Multi-Source GeoRAG

```
Q: Will Taiwan tensions affect Nvidia?
A: ## GeoRAG Analysis
   Intent: multi_source (confidence: 0.85)
   Sources: historical_events, knowledge_graph, market_data

   Historical Analogs:
   - Taiwan Straits Crisis (2022): SOX -15%, recovered in 2 weeks
   - US-China Trade War (2018-2019): Tech sector -10% over 3 months

   Graph Paths:
   Taiwan[manufactures]→Semiconductors[essential for]→Nvidia[supply chain]→Global Tech

   Market Context:
   Semiconductor stocks typically decline 10-15% during Taiwan tensions
   but recover within 1-2 months post-crisis.
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
├── main.py                          # FastAPI entry point (77 lines)
├── run.bat                          # One-click launcher
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── pytest.ini                       # Test config
├── rag/                             # *** GeoRAG Pipeline System ***
│   ├── __init__.py                  # Top-level exports + seed_knowledge_base()
│   ├── chunking/                    # 5 text splitting strategies
│   ├── embeddings/                  # BGE-M3 embedding model
│   ├── vectorstore/                 # Qdrant vector DB client
│   ├── ingestion/                   # Document ingestion pipeline
│   ├── retrievers/                  # 5 specialized retrievers
│   ├── rerankers/                   # BGE cross-encoder reranker
│   ├── historical_memory/           # Event similarity engine (26 events)
│   ├── graph_retrieval/             # Knowledge graph (13 entities)
│   ├── geo_rag/                     # Orchestration layer
│   └── pipelines/                   # 6 production pipelines
├── app/
│   ├── config.py                    # Pydantic settings from env
│   ├── models.py                    # All Pydantic models
│   ├── agents/                      # 11 agent modules
│   │   ├── intent_router.py         # Query classifier (8 intents)
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
│   │   └── data.py                  # 53 countries, 40 routes, 69 ports
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
│   │   ├── embeddings.py            # BGE-M3 embeddings (legacy)
│   │   ├── vector_store.py          # Qdrant client (legacy)
│   │   └── retriever.py             # RAG pipeline (legacy)
│   ├── utils/
│   │   ├── metrics.py               # Volatility, momentum, Sharpe
│   │   └── constants.py             # Sector→ETF map, keywords
│   └── workflow/
│       └── graph.py                 # LangGraph StateGraph
├── tests/
│   ├── test_chatbot.py              # 19 legacy tests
│   ├── test_rag_pipelines.py        # Comprehensive RAG module tests
│   ├── quick_test_rag.py            # Offline-compatible smoke tests
│   └── test_rag_advanced.py         # End-to-end pipeline integration tests
├── scripts/
│   └── download_model.py            # BGE-M3 download script
├── memory_store/                    # Long-term memory JSON files
└── .cache/                          # HuggingFace model cache
```

---

## Testing

```powershell
# Legacy tests (19 tests)
python -m pytest tests/test_chatbot.py -v

# RAG pipeline tests
python tests/quick_test_rag.py
python tests/test_rag_advanced.py
```

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
- `transformers` — BGE reranker (optional)

All external services (Ollama, Qdrant, Neo4j, PostgreSQL) are optional. The system works fully offline with MockLLM fallback and in-memory data stores.
