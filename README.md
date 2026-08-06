# Generic CZ Portal Automation

This repository is being rebuilt as a deterministic, scraping-first automation
pipeline for generic CZ portals.

The active implementation contains the immutable evidence contract, durable
artifact store, pre-navigation Playwright recorder, deterministic page
snapshotter, action safety policy, bounded state-graph explorer, and production
crawl runner. It now also includes the deterministic normalization boundary
that turns immutable crawl evidence into cited, LLM-ready portal knowledge. The
runner can explicitly export a private authenticated session for reuse without
mixing raw session material into crawl evidence.

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

Deterministic browser recording, state extraction, graph exploration, the
authenticated production entrypoint, and portal-knowledge normalization are
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
  runner.py          # Typed auth/session boundary and production orchestration
  cli.py             # Secret-safe validation and crawl command
  __main__.py        # python -m scraper entrypoint
knowledge/
  models.py          # Strict normalized tables, testcases, routes, and coverage
  builder.py         # Generic semantic extraction from immutable crawl evidence
  exporter.py        # Atomic audit JSON and compact testcase JSONL export
  cli.py             # Crawl-to-knowledge command
  __main__.py        # python -m knowledge entrypoint
tests/
  test_models.py
  test_artifact_store.py
  test_redaction.py
  test_browser_integration.py
  test_extractor.py
  test_snapshot_integration.py
  test_actions.py
  test_explorer_integration.py
  test_runner.py
  test_runner_integration.py
  test_knowledge_builder.py
```

## Development

```bash
python -m pip install -e ".[dev]"
playwright install chromium
python -m pytest
CZ_RUN_BROWSER_TESTS=1 python -m pytest -q \
  tests/test_browser_integration.py tests/test_snapshot_integration.py \
  tests/test_explorer_integration.py tests/test_runner_integration.py
python -m py_compile scraper/*.py knowledge/*.py
```

## Running a crawl

Validate configuration without starting a browser or creating a run:

```bash
python -m scraper --url "https://portal.example.test/start" --validate-only
```

Run an anonymous or externally authenticated portal session:

```bash
python -m scraper \
  --url "https://portal.example.test/start" \
  --run-id portal-baseline
```

For interactive SSO, no username or password is accepted on the command line.
The headed browser pauses until login is completed manually. A caller-provided
ready selector is optional and is never hardcoded by the crawler:

```bash
python -m scraper \
  --url "https://portal.example.test/start" \
  --auth manual --headed \
  --ready-selector "[data-portal-ready]"
```

To create a reusable session during that manual login, choose an explicit
ignored output under `artifacts/`:

```bash
python -m scraper \
  --url "https://portal.example.test/start" \
  --run-id portal-login \
  --auth manual --headed \
  --save-storage-state artifacts/sessions/portal.json
```

Press Enter only after the authenticated portal page is visible. The exported
file contains raw cookies and Web Storage credentials, is written with owner-only
permissions on POSIX systems, and is not part of the content-addressed evidence
store or manifest. It must be treated like a password. An existing output is
never replaced unless `--overwrite-storage-state` is supplied explicitly.

An existing Playwright storage-state file can instead be injected at runtime:

```bash
python -m scraper \
  --url "https://portal.example.test/start" \
  --auth storage_state \
  --storage-state artifacts/sessions/portal.json
```

Storage-state input and output paths are not written to the manifest or printed
by `--validate-only`. The manifest records only secret-free behavior, including
whether external session state was configured and a hash—not the text—of an
optional readiness selector. Runtime URL values matching the redaction
vocabulary are redacted before the root URL enters evidence. Normal crawl
storage artifacts contain hashes, not cookie or Web Storage plaintext. The
explicit reusable session export is the intentional exception. Screenshots and
raw traces can still show sensitive rendered data and must be protected.

Run IDs are exclusive by default. `--existing-run return_completed` verifies
integrity and returns an already completed compatible run without launching a
browser. `--existing-run new_attempt` creates `RUN_ID-attempt-N` without
overwriting prior evidence. Resuming a partial live browser frontier is
intentionally unsupported because credentials and live page state are not
persisted; start a new attempt instead.

The production path has been verified end to end with local HTTP fixtures and
real Chromium, including authenticated-session export followed by reuse in a fresh
browser context. Authenticated state extraction has also been smoke-tested on an
actual CZ environment. Deeper live traversal remains deferred until that portal
contains representative test cases.

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
enter the evidence store. This includes OTP/CSRF name-value patterns and
security-related HTML metadata. Trace archives are retained as explicitly
unredacted raw evidence and must be handled as sensitive artifacts.

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
Nested visual nodes inside one activatable control remain in the evidence but
are skipped as duplicate click targets. Broad delegated-listener containers with
multiple interactive descendants are also retained but not clicked because no
single deterministic target can be inferred.

Before each safe sibling action, the explorer restores the root cookies and
root-state Web Storage, reloads the root page, then replays the safe locator
path to the parent. Detached iframe predecessors are excluded so frame paths and
state fingerprints remain stable across reloads. Popup creation is observed
from before the triggering action so initial execution and replay select the
same resulting page. Review-required actions are
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

## Portal-knowledge normalization

After a completed bounded crawl, normalize its evidence into the stable input
contract for repository retrieval and the later LLM phase:

```bash
python -m knowledge --run-id portal-baseline
```

By default this reads `artifacts/crawls/portal-baseline` and writes an isolated
package beneath `artifacts/knowledge/portal-baseline`:

- `portal_knowledge.json` contains audit-grade normalized tables, testcase
  fields, dependencies, controls, modal descriptions, routes, network
  observations, conflicts, limitations, and immutable evidence citations.
- `testcases.jsonl` contains one compact, deterministic LLM input record per
  testcase while retaining source URLs and evidence state IDs.
- `manifest.json` records file hashes, sizes, counts, and completion gates.

Extraction is portal-agnostic: it uses semantic table headers, structural row
relationships, dialog ancestry, and recorded action metadata rather than CZ
text or CSS selectors. `source_bounded_complete` reports whether the entire
configured crawl frontier completed. `testcase_context_complete` is a separate
gate that is true only when the portal's declared total equals the normalized
testcase count and every testcase has non-conflicting description evidence.

A partial crawl can be inspected only through an explicit diagnostic export:

```bash
python -m knowledge \
  --run-id portal-partial \
  --allow-incomplete
```

Such a package preserves its source limitation and must not be treated as full
portal coverage. `--skip-integrity-check` is also intended only for diagnostics;
normal handoffs verify the crawl store before normalization.

The previous agentic implementation is not part of the active import graph. A
local recoverable copy is stored under `.deprecated/`, which is intentionally
ignored by Git. The original tracked versions remain available in Git history.
