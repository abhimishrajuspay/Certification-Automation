# Generic CZ Portal Automation

This repository is being rebuilt as a deterministic, scraping-first automation
pipeline for generic CZ portals.

The active implementation contains the immutable evidence contract, durable
artifact store, pre-navigation Playwright recorder, deterministic page
snapshotter, action safety policy, bounded state-graph explorer, and production
crawl runner. It now also includes the deterministic normalization boundary
that turns immutable crawl evidence into cited portal knowledge, plus bounded
repository and MCP grounding for every normalized testcase. The runner can
explicitly export a private authenticated session for reuse without mixing raw
session material into crawl evidence. Phase 9 compiles that cited context into
strict, review-aware HTTP execution specifications through a LiteLLM proxy.
Phase 10 deterministically renders execution-ready specifications as a Postman
Collection v2.1 plus a secret-empty environment template.

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
authenticated production entrypoint, portal-knowledge normalization, cited
repository/MCP retrieval, and constrained LLM synthesis are active. Postman
Collection v2.1 generation is also active and remains LLM-free by design.

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
grounding/
  models.py          # Strict repository/MCP citations and coverage gates
  repository.py      # Bounded, redacted, line-cited local retrieval
  mcp.py             # Async JSON-RPC client with read-only tool allowlist
  builder.py         # Grouped testcase retrieval and shared snippet assembly
  exporter.py        # Atomic audit JSON and LLM-ready JSONL export
  cli.py             # Knowledge-to-grounding command
  __main__.py        # python -m grounding entrypoint
synthesis/
  models.py          # Strict HTTP request, assertion, review, and audit models
  client.py          # Bounded OpenAI-compatible LiteLLM proxy client
  builder.py         # Semantic batching, prompting, retries, and citation checks
  exporter.py        # Atomic audit/JSONL export and resumable checkpoints
  cli.py             # Grounding-to-execution-spec command
  __main__.py        # python -m synthesis entrypoint
postman/
  models.py          # Strict Collection v2.1, environment, and audit contracts
  builder.py         # Dependency ordering, variables, scripts, and assertions
  exporter.py        # Atomic collection/environment/report export
  cli.py             # Execution-spec-to-Postman command
  __main__.py        # python -m postman entrypoint
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
  test_grounding.py
  test_synthesis.py
  test_postman.py
```

## Development

```bash
python -m pip install -e ".[dev]"
playwright install chromium
python -m pytest
CZ_RUN_BROWSER_TESTS=1 python -m pytest -q \
  tests/test_browser_integration.py tests/test_snapshot_integration.py \
  tests/test_explorer_integration.py tests/test_runner_integration.py
python -m py_compile \
  scraper/*.py knowledge/*.py grounding/*.py synthesis/*.py postman/*.py
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

## Repository and MCP grounding

Ground a normalized package against the integration repository and the MCP
documentation server before sending anything to an LLM:

```bash
export CZ_MCP_URL="http://10.200.3.108:8000/mcp"
python -m grounding \
  --run-id portal-baseline \
  --repo-path /path/to/integration-repository
```

The repository scanner reads only an explicit extension allowlist for schemas,
templates, examples, and documentation. It does not follow symlinks, skips
hidden/build/runtime directories, enforces per-file and total byte limits, and
redacts sensitive name/value patterns before excerpts are retained. Citations
contain a relative path, raw file SHA-256, line range, and excerpt digest; the
absolute repository path is not exported.

The MCP client performs a standard JSON-RPC initialize and tool-discovery
handshake. Automatic grounding can invoke only the advertised `search_docs` and
`search_documents` tools. Testcases are grouped by semantic API identity so a
portal with many scenarios for one API does not issue one remote query per row.
Every retained MCP excerpt records the server/protocol identity, tool,
argument/response digests, retrieval time, and document/chunk references.
Payload-generation, sandbox execution, validation, and other side-effecting or
generative MCP tools are never called by this phase.

The default output under `artifacts/grounding/<run-id>/` contains:

- `grounding.json`, the audit-grade package with shared content-addressed
  snippets and per-testcase references;
- `grounded_testcases.jsonl`, one self-contained context record per testcase for
  the next structured LLM phase; and
- `manifest.json`, containing output hashes and the `grounding_complete` gate.

Grounding is complete only when Phase 7 testcase context is complete, every
testcase has repository or MCP context, and configured MCP retrieval completed.
Repository-only diagnostics can use `--no-mcp`. Incomplete output returns exit
code 2 unless `--allow-incomplete-grounding` is explicitly supplied.

## Constrained execution-spec synthesis

First validate the Phase 8 package and inspect the number and size of planned
model calls. This performs no network request and requires no model credential:

```bash
python -m synthesis --run-id portal-baseline --plan-only
```

Configure an OpenAI-compatible LiteLLM proxy. Supply the API key through the
environment or a secret manager; it is never accepted as a CLI value or written
to an artifact:

```bash
export CZ_LITELLM_URL="https://litellm.example.test"
export CZ_LITELLM_MODEL="your-proxy-model-name"
# Inject CZ_LITELLM_API_KEY with your shell or secret manager.

python -m synthesis \
  --run-id portal-baseline \
  --maximum-cases-per-batch 8 \
  --concurrency 2
```

Use `--no-api-key` only for a trusted proxy that authenticates outside the
request. If the selected model cannot accept JSON Schema response formatting,
use `--response-format json_object`; the same Pydantic validation still applies
after the response. A stopped or partially failed run can continue without
repeating completed batches:

```bash
python -m synthesis --run-id portal-baseline --resume
```

The default output under `artifacts/synthesis/<run-id>/` contains:

- `synthesis.json`, the audit package with secret-free model-call hashes,
  validation history, token totals, coverage, and every execution spec;
- `execution_specs.jsonl`, one strict request/assertion specification per
  successfully synthesized testcase;
- `checkpoint.json`, atomically updated after each completed batch; and
- `manifest.json`, with final output hashes and separate `synthesis_complete`
  and `execution_ready` gates.

Each testcase is classified as `ready`, `needs_review`, or `blocked`. A complete
synthesis may still be not execution-ready when the cited sources do not contain
enough facts. Ready requests cannot contain fixed origins, plaintext sensitive
headers, invalid JSON/XML, redaction markers, invented snippet IDs, or changed
dependencies. Every request/assertion placeholder has a typed environment,
portal-field, cited-literal, generated, or dependency binding; copied portal
values are checked against the Phase 8 input. Raw model responses and prompts
are deliberately not persisted.

The LLM—not MCP—makes these decisions. Phase 8 MCP tools only retrieve cited
documentation. Phase 9 sends portal facts plus those excerpts to the configured
LiteLLM model, which proposes the HTTP method/path, payload, typed variables,
assertions, confidence, and ready/review/blocked disposition. Local validators
reject unsupported or invented output before anything reaches Postman.

## Postman Collection v2.1 generation

After Phase 9 has produced `execution_specs.jsonl`, inspect deterministic render
coverage without writing files:

```bash
python -m postman --run-id portal-baseline --plan-only
```

Generate the final package only when Phase 9 reports `execution_ready: true`:

```bash
python -m postman --run-id portal-baseline
```

The default output under `artifacts/postman/<run-id>/` contains:

- `postman_collection.json`, with requests in dependency order, typed variable
  setup, dependency guards, response assertions, and parent-response extraction;
- `postman_environment.json`, containing an empty `base_url` plus empty runtime
  credential variables marked as secrets;
- `postman_report.json`, with rendered/skipped testcase coverage and reasons;
  and
- `manifest.json`, containing content hashes and `generation_complete`.

The renderer never asks MCP or an LLM to make another decision. It translates
only validated Phase 9 fields. A child request is skipped unless all parents
passed, dependency cycles are rejected, and environment secrets are never copied
into generated files. For diagnostics, `--allow-partial` renders only the ready,
dependency-closed subset and records every omission; it never claims complete
generation.

The previous agentic implementation is not part of the active import graph. A
local recoverable copy is stored under `.deprecated/`, which is intentionally
ignored by Git. The original tracked versions remain available in Git history.
