# Project instructions

## Current purpose

Build a deterministic, scraping-first automation pipeline for generic CZ
portals. The browser crawler must collect auditable evidence before any LLM,
repository retrieval, MCP, testcase synthesis, or Postman generation occurs.

## Active phase

Phase 1 is the immutable evidence contract in `scraper/models.py`.

Active code must not import from `.deprecated/`. The local legacy archive is
for recovery and comparison only and is intentionally ignored by Git.

## Architecture rules

- Python 3.10+ with fully typed signatures.
- Pydantic v2 models for serialized evidence.
- Evidence models are frozen, reject unknown fields, and use tuple collections.
- Timestamps must be timezone-aware.
- Large evidence is represented by content-addressed artifact references.
- Plaintext secrets must never be retained in values marked as redacted.
- The crawl graph and future testcase dependency graph are separate concepts.
- Browser collection remains deterministic and LLM-free.
- Runtime output belongs under `artifacts/` and stays out of Git.

## Validation

- Focused tests: `venv/bin/python3 -m pytest -q tests/test_models.py`
- Compile check:
  `venv/bin/python3 -m py_compile scraper/models.py scraper/__init__.py tests/test_models.py`
- Full checks when development tools are installed: `black --check .`,
  `flake8 .`, and `mypy .`

## Safety

- Preserve `.env`; never copy it into artifacts, archives, logs, or chat.
- Treat browser cookies, authorization headers, request bodies, screenshots,
  and storage state as potentially sensitive.
- Do not restore or reactivate legacy modules unless the user explicitly asks.
