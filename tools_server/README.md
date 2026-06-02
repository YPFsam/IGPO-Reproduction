# IGPO Tool Server - Web Search

Lightweight web search tool server for IGPO training.

## Quick Start

1. Choose Serper, Bing, mock mode, or a local Tantivy server.

2. Edit `config.yaml`:
```yaml
serper_api_key: "your_api_key_here"
```

3. Run training - the tool server will be used automatically.

## Configuration

```yaml
# config.yaml
search_engine: "google"     # or "bing"
search_top_k: 10            # results per query
serper_api_key: "xxx"       # Serper API key
```

Per-run environment variables override `config.yaml` without modifying tracked
files:

```bash
IGPO_MOCK_SEARCH=true bash train_1gpu_mock.sh

IGPO_MOCK_SEARCH=false \
IGPO_SEARCH_ENGINE=local \
IGPO_LOCAL_SEARCH_URL=http://localhost:8890/search \
  bash train_1gpu_mock.sh local

IGPO_SERPER_API_KEY='<serper-key>' \
  bash train_1gpu_mock.sh online
```

## Supported Search Engines

| Engine | Provider | Notes |
|--------|----------|-------|
| Google | Serper API | 2,500 free searches |
| Bing | Azure | Pay as you go |
| Local | Tantivy HTTP server | Compatible with the DR-Venus local-search adapter |
| Mock | Built-in deterministic snippets | Pipeline checks only |

## Files

```
tools_server/
├── config.yaml      # Configuration
├── handler.py       # Web search handler
├── tools.py         # Tool definition
├── util.py          # MessageClient
└── search/
    └── search_api.py  # Search implementations
```
