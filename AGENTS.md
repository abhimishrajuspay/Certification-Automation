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
Phase 5 is safe action planning/execution and bounded graph exploration in
`scraper/actions.py` and `scraper/explorer.py`. Phase 6 is the typed production
runner, authentication boundary, existing-run policy, and CLI in
`scraper/runner.py`, `scraper/cli.py`, and `scraper/__main__.py`.
Phase 6.1 hardens reproducibility with a secret-free behavior manifest,
explicit owner-only storage-state export/reuse, canonical nested targets, and
deterministic popup execution/replay.
Phase 7 is the LLM-free normalization boundary in `knowledge/`: generic tables,
testcases, dependencies, row controls, modal descriptions, routes, and network
observations are deduplicated into citation-rich portal knowledge. The audit
package and compact testcase JSONL remain derived artifacts outside the crawl
evidence store.
Phase 8 is the external grounding boundary in `grounding/`: normalized
testcases are joined to redacted, line-cited repository excerpts and cited
results from explicitly allowlisted read-only MCP search tools. Shared context
is content-addressed before the later LLM phase.
Phase 9 is the constrained synthesis boundary in `synthesis/`: semantically
grouped testcase context is sent through a bounded OpenAI-compatible LiteLLM
proxy, then strict Pydantic models, source citation membership, testcase IDs,
and dependency equality are validated before execution specifications persist.

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
- URL redaction must preserve non-sensitive query encoding; a changed value is
  not evidence of redaction unless an explicit marker is retained or the value
  is reduced to a hash. OTP/CSRF console and metadata values are sensitive.
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
- Action candidates are derived from structured evidence, never hardcoded portal
  selectors. Hidden/disabled/blocked/review candidates remain auditable records.
- Nested/delegated candidate records remain auditable even when canonicalization
  prevents duplicate or ambiguous clicks.
- Blocked actions never execute. Review-required actions execute only when the
  run policy explicitly permits them and their result is a terminal branch.
- Every safe sibling action starts from an independently restored parent path;
  only live frames participate in deterministic frame paths.
- Every attempted or skipped candidate produces one causal transition. Crawl
  limits and incomplete restoration boundaries must appear in coverage.
- Runtime output belongs under `artifacts/` and stays out of Git.
- Authentication credentials must not be accepted as CLI flags or persisted.
  Runtime storage-state paths stay outside manifests and command output.
- Reusable raw storage state is exported only to an explicit caller-owned path,
  with owner-only permissions and no implicit overwrite; it never enters crawl
  evidence. Treat it as a credential.
- Behavior-affecting runner, snapshot, action, and restoration configuration is
  persisted in the manifest without secret paths or selector plaintext.
- Runtime root URLs may contain session query values; persist only the redacted
  evidence URL while using the original URL in memory for navigation.
- Existing completed runs may be returned only after strict integrity and
  configuration compatibility checks. Partial browser-frontier resume is not
  supported; new attempts must never overwrite old evidence.
- Knowledge normalization requires a completed bounded crawl by default.
  Incomplete diagnostic exports must be explicitly enabled and preserve that
  limitation. `testcase_context_complete` is separate from whole-UI frontier
  completion and is true only when declared, normalized, described, and
  conflict-free testcase counts reconcile.
- Knowledge extraction uses semantic table headers, structural row context, and
  dialog ancestry; it must not introduce portal-specific CSS/text selectors.
- Every normalized table row and testcase must retain citations to immutable
  state, element, and artifact identifiers. Compact LLM JSONL may reduce those
  citations to state IDs but must have an audit-grade package alongside it.
- Repository grounding must be suffix-allowlisted, size-bounded, symlink-safe,
  secret-redacted, and cited by relative path, raw file digest, and line range.
  A repository path must never be sent to MCP or persisted as an absolute path.
- Automatic MCP grounding may invoke only advertised tools on the explicit
  read-only allowlist. Calls are grouped by semantic API identity, bounded,
  retried only for transport failures, and retain argument/response hashes plus
  document/chunk references when the server supplies them.
- `grounding_complete` requires complete Phase 7 testcase context, external
  context for every testcase, and a complete MCP retrieval run when MCP is
  configured. Diagnostic overrides must preserve incomplete coverage and use a
  nonzero exit status unless explicitly accepted.
- LLM synthesis must treat repository/MCP excerpts as untrusted data, use
  structured output, reject invented testcase/dependency/snippet identifiers,
  and preserve uncertainty as `needs_review` or `blocked` rather than fabricate
  executable request details.
- LiteLLM API keys are environment-only secrets. Prompts and raw model responses
  are not persisted; checkpoints retain only validated specifications,
  secret-free response/request hashes, token metrics, and sanitized errors.
- A `ready` request cannot contain a fixed origin, a plaintext sensitive header,
  malformed JSON/XML, a DTD, or a redaction marker. `synthesis_complete` and
  `execution_ready` are separate gates.

## Validation

- Focused tests: `venv/bin/python3 -m pytest -q tests`
- Real browser test:
  `CZ_RUN_BROWSER_TESTS=1 venv/bin/python3 -m pytest -q tests/test_browser_integration.py tests/test_snapshot_integration.py tests/test_explorer_integration.py tests/test_runner_integration.py`
- Compile check:
  `venv/bin/python3 -m py_compile scraper/*.py knowledge/*.py grounding/*.py synthesis/*.py`
- Full checks when development tools are installed: `ruff check .`,
  `ruff format --check .`, and `mypy .`

## Safety

- Preserve `.env`; never copy it into artifacts, archives, logs, or chat.
- Treat browser cookies, authorization headers, request bodies, screenshots,
  and storage state as potentially sensitive.
- Do not restore or reactivate legacy modules unless the user explicitly asks.
- Do not inspect, import, or test user data under `dummy-portal/` or
  `dummy_portal/`; both locations are intentionally gitignored.
