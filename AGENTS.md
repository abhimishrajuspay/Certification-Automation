# Project instructions

## Current purpose

Build a deterministic, scraping-first automation pipeline for generic CZ
portals. The browser crawler must collect auditable evidence before any LLM,
repository retrieval, MCP, testcase synthesis, or Postman generation occurs.

## Active foundation

Phase 1 is the immutable evidence contract in `scraper/models.py`. Phase 2 is
the append-only, content-addressed store in `scraper/artifact_store.py`. Phase 3
is the pre-navigation Playwright lifecycle and browser recorder in
`scraper/browser.py`, `scraper/recorder.py`, and `scraper/redaction.py`. Phase 4
is deterministic page-state extraction in `scraper/extractor.py`.

Active code must not import from `.deprecated/`. The local legacy archive is
for recovery and comparison only and is intentionally ignored by Git.

## Architecture rules

- Python 3.10+ with fully typed signatures.
- Pydantic v2 models for serialized evidence.
- Evidence models are frozen, reject unknown fields, and use tuple collections.
- Timestamps must be timezone-aware.
- Large evidence is represented by content-addressed artifact references.
- Structured evidence is persisted in homogeneous sequence-checked JSONL
  streams; blobs are immutable and addressed by SHA-256.
- Only manifests and checkpoints may use atomic replacement.
- Plaintext secrets must never be retained in values marked as redacted.
- The crawl graph and future testcase dependency graph are separate concepts.
- Browser collection remains deterministic and LLM-free.
- Browser listeners and page init scripts must be installed before navigation.
- Browser actions must run inside an action scope until their observable effects
  settle so network and event evidence keeps its causal `action_id`.
- Embedded HAR text must be redacted before persistence; raw Playwright traces
  are explicitly unredacted sensitive artifacts.
- State extraction must be interaction-free and commit the `StateSnapshot` only
  after every referenced frame/state artifact is durable.
- Element IDs and state fingerprints are deterministic; capture sequence and
  runtime UUIDs must not affect state identity.
- Live form/storage values are hashed, not persisted as plaintext.
- Browser limitations such as closed shadow roots, delegated listeners, canvas,
  pseudo-elements, and off-DOM virtualized content must be reported honestly.
- Runtime output belongs under `artifacts/` and stays out of Git.

## Validation

- Focused tests: `venv/bin/python3 -m pytest -q tests`
- Real browser test:
  `CZ_RUN_BROWSER_TESTS=1 venv/bin/python3 -m pytest -q tests/test_browser_integration.py tests/test_snapshot_integration.py`
- Compile check:
  `venv/bin/python3 -m py_compile scraper/*.py`
- Full checks when development tools are installed: `ruff check .`,
  `ruff format --check .`, and `mypy .`

## Safety

- Preserve `.env`; never copy it into artifacts, archives, logs, or chat.
- Treat browser cookies, authorization headers, request bodies, screenshots,
  and storage state as potentially sensitive.
- Do not restore or reactivate legacy modules unless the user explicitly asks.
