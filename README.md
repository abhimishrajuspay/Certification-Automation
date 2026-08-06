# Generic CZ Portal Automation

This repository is being rebuilt as a deterministic, scraping-first automation
pipeline for generic CZ portals.

The active implementation contains the immutable evidence contract, durable
artifact store, pre-navigation Playwright recorder, deterministic page
snapshotter, action safety policy, and bounded state-graph explorer.

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

Deterministic browser recording, state extraction, and graph exploration are
active. LLM synthesis, MCP retrieval, and Postman generation will be added in
later phases without coupling them to evidence persistence.

## Project layout

```text
scraper/
  __init__.py
  models.py          # Immutable Pydantic evidence models
  artifact_store.py  # Content-addressed blobs and append-only JSONL streams
  redaction.py       # Deterministic URL/header/body/HAR secret redaction
  recorder.py        # Context-wide browser, network, and mutation recorder
  browser.py         # Pre-navigation Playwright lifecycle and trace/HAR ingest
  extractor.py       # Frame/DOM/element/storage/screenshot state snapshots
  actions.py         # Candidate derivation, safety policy, replay, and effects
  explorer.py        # Parent restoration, bounded frontier, coverage/finalize
tests/
  test_models.py
  test_artifact_store.py
  test_redaction.py
  test_browser_integration.py
  test_extractor.py
  test_snapshot_integration.py
  test_actions.py
  test_explorer_integration.py
```

## Development

```bash
python -m pip install -e ".[dev]"
playwright install chromium
python -m pytest
CZ_RUN_BROWSER_TESTS=1 python -m pytest -q \
  tests/test_browser_integration.py tests/test_snapshot_integration.py \
  tests/test_explorer_integration.py
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

The state extractor waits for a bounded DOM-quiet window without clicking,
then traverses the main document, child frames, and open shadow roots. It
captures every render-relevant DOM element, including hidden and disabled
controls, with semantic names, labels, text, attributes, live-state hashes,
geometry, form/table/section context, select options, stable locator candidates,
and interaction signals. Listener registration is observed from document start,
so non-native elements wired with `addEventListener` are discoverable. Each
frame receives content-addressed DOM, element, and accessibility-oriented
artifacts; each state also captures a full-page screenshot and a value-hashed
storage summary. Identical consecutive states produce the same fingerprint and
state ID.

Browser platform boundaries remain explicit: closed shadow roots are not
inspectable; delegated event listeners cannot always be mapped to one child;
canvas internals and pseudo-elements are represented visually by screenshots;
and virtualized content that is not yet in the DOM requires later scrolling by
the state-graph explorer. Screenshots and raw network/trace evidence can contain
sensitive rendered data and must be handled accordingly.

## State-graph exploration

The action planner derives candidates from semantic element evidence rather
than portal text selectors. It records hidden, disabled, cross-origin,
destructive, and review-required controls even when policy prevents execution.
Safe clicks, toggles, selects, registered hover/double-click handlers, iframe
controls, popups, and bounded page scrolling can be executed. Each action runs
inside its recorder scope, captures the resulting state, correlates browser and
network evidence, and persists a causal transition with normalized effects.

Before each safe sibling action, the explorer restores the root cookies and
root-state Web Storage, reloads the root page, then replays the safe locator
path to the parent. Detached iframe predecessors are excluded so frame paths and
state fingerprints remain stable across reloads. Review-required actions are
disabled by default; when explicitly enabled, their immediate result is
captured once and the branch is terminal so a potentially external side effect
is never replayed automatically. Destructive and disallowed cross-origin
actions are never executed.

Depth, state, action, and runtime limits are taken from the run manifest. The
final manifest, checkpoint, transition journal, and `CoverageReport` state
whether the bounded frontier was exhausted and why each unexecuted action was
skipped. Runtime restoration cannot undo server-side effects or reset storage
for origins absent from the root frame tree, so only policy-approved safe
actions are eligible for automatic replay.

The previous agentic implementation is not part of the active import graph. A
local recoverable copy is stored under `.deprecated/`, which is intentionally
ignored by Git. The original tracked versions remain available in Git history.
