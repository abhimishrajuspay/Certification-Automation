# Generic CZ Portal Automation

This repository is being rebuilt as a deterministic, scraping-first automation
pipeline for generic CZ portals.

The active implementation contains the immutable evidence contract, durable
artifact store, and a Playwright lifecycle that attaches evidence recording
before the first navigation. These components are shared by the future
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

Deterministic browser event and network capture is active. State extraction,
graph exploration, LLM synthesis, MCP retrieval, and Postman generation will
be added in later phases without coupling them to evidence persistence.

## Project layout

```text
scraper/
  __init__.py
  models.py          # Immutable Pydantic evidence models
  artifact_store.py  # Content-addressed blobs and append-only JSONL streams
  redaction.py       # Deterministic URL/header/body/HAR secret redaction
  recorder.py        # Context-wide browser, network, and mutation recorder
  browser.py         # Pre-navigation Playwright lifecycle and trace/HAR ingest
tests/
  test_models.py
  test_artifact_store.py
  test_redaction.py
  test_browser_integration.py
```

## Development

```bash
python -m pip install -e ".[dev]"
playwright install chromium
python -m pytest
CZ_RUN_BROWSER_TESTS=1 python -m pytest -q tests/test_browser_integration.py
python -m py_compile scraper/*.py
```

## Evidence storage

Each scrape run owns an isolated directory beneath a caller-provided artifact
root. Large evidence is deduplicated by SHA-256 under `blobs/`. States, actions,
transitions, browser events, network exchanges, and coverage reports are stored
in separate sequence-checked JSONL streams. The run manifest and recovery
checkpoint use atomic replacement; interrupted trailing JSONL writes can be
detected and explicitly repaired during reopen.

The browser manager opens an instrumented `about:blank` page, installs the DOM
mutation hook, attaches context and page listeners, and only then permits
navigation. It records requests/responses, text bodies, console and page errors,
frames, dialogs, popups, downloads, file choosers, workers, WebSockets,
navigations, lifecycle events, DOM mutations, HAR, and trace references. Action
scopes correlate asynchronous effects with the interaction that caused them.
Sensitive header, URL, body, console, and HAR fields are redacted before they
enter the evidence store. Trace archives are retained as explicitly unredacted
raw evidence and must be handled as sensitive artifacts.

The previous agentic implementation is not part of the active import graph. A
local recoverable copy is stored under `.deprecated/`, which is intentionally
ignored by Git. The original tracked versions remain available in Git history.
