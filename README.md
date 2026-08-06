# Generic CZ Portal Automation

This repository is being rebuilt as a deterministic, scraping-first automation
pipeline for generic CZ portals.

The active implementation currently contains Phase 1: the immutable evidence
contract shared by the future browser recorder, snapshotter, state explorer,
artifact store, and coverage reporter.

## Active architecture

```text
Instrumented browser
  -> deterministic state-graph crawler
  -> immutable crawl evidence
  -> normalized portal knowledge
  -> repository and MCP retrieval
  -> structured LLM synthesis
  -> Postman Collection v2.1
```

Only the immutable crawl evidence layer is implemented. Browser automation,
LLM synthesis, MCP retrieval, and Postman generation will be added in later
phases without coupling them to the evidence models.

## Project layout

```text
scraper/
  __init__.py
  models.py          # Immutable Pydantic evidence models
tests/
  test_models.py     # Contract validation and serialization tests
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m py_compile scraper/models.py scraper/__init__.py tests/test_models.py
```

The previous agentic implementation is not part of the active import graph. A
local recoverable copy is stored under `.deprecated/`, which is intentionally
ignored by Git. The original tracked versions remain available in Git history.
