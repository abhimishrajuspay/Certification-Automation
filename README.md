# Generic CZ Portal Automation

This repository is being rebuilt as a deterministic, scraping-first automation
pipeline for generic CZ portals.

The active implementation contains the immutable evidence contract and its
durable artifact store. These are shared by the future browser recorder,
snapshotter, state explorer, and coverage reporter.

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

Browser automation, LLM synthesis, MCP retrieval, and Postman generation will
be added in later phases without coupling them to evidence persistence.

## Project layout

```text
scraper/
  __init__.py
  models.py          # Immutable Pydantic evidence models
  artifact_store.py  # Content-addressed blobs and append-only JSONL streams
tests/
  test_models.py
  test_artifact_store.py
```

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m py_compile scraper/models.py scraper/artifact_store.py scraper/__init__.py
```

## Evidence storage

Each scrape run owns an isolated directory beneath a caller-provided artifact
root. Large evidence is deduplicated by SHA-256 under `blobs/`. States, actions,
transitions, browser events, network exchanges, and coverage reports are stored
in separate sequence-checked JSONL streams. The run manifest and recovery
checkpoint use atomic replacement; interrupted trailing JSONL writes can be
detected and explicitly repaired during reopen.

The previous agentic implementation is not part of the active import graph. A
local recoverable copy is stored under `.deprecated/`, which is intentionally
ignored by Git. The original tracked versions remain available in Git history.
