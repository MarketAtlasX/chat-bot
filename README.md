# MarketAtlas Chat

AI-Powered Geopolitical Trading Intelligence Chatbot.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

Open http://localhost:8000

## API

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Why is oil rising today?"}'
```

## Architecture

| Layer | Technology |
|-------|-----------|
| LLM | Ollama (Qwen / Mistral / Llama) + Mock fallback |
| Orchestration | LangGraph |
| Vector Store | Qdrant + BGE-M3 |
| Graph DB | Neo4j |
| SQL DB | PostgreSQL |
| API | FastAPI + WebSocket |

## Agents

- Intent Router → classifies queries (NEWS, MARKET, IMPACT, etc.)
- News Agent → retrieves and summarizes geopolitical news
- Market Agent → analyzes financial market data
- Impact Agent → scores geopolitical risk
- Graph Agent → queries knowledge graph relationships
- Forecast Agent → probability-weighted scenario forecasting
- Recommendation Agent → trade suggestions
- Simulation Agent → what-if geopolitical simulations
- Report Agent → structured intelligence reports
- Debate Agent → multi-analyst perspective synthesis

## Tests

```powershell
python -m pytest tests/test_chatbot.py -v
```
