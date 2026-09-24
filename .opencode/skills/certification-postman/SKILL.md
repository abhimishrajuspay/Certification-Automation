---
name: certification-postman
description: Plan, generate, and audit a Postman Collection v2.1 from a certification testcase CSV using evidence-backed endpoint mappings, prerequisite chains, MCP/API specs, and an optional reference collection. Use for complete certification suites, not one-off requests or live environment mutation.
---

# Certification Postman

Deterministic build path: **CSV → endpoint mapping (mapping.json) → build → audit → ship.**
Planning, semantic grouping, dependency classification, evidence selection, and
`mapping.json` authoring require agent judgment; build and static audit are
scripted. Work family-by-family so evidence amortizes across cases.

## Operating modes and phase gate

Determine the requested stopping point before acting:

1. **Planning/inventory only (read-only):** validate and normalize the input,
   group cases, classify every dependency, discover endpoint candidates, and
   report missing/ambiguous families. Write the planning artifacts listed
   below and STOP. Do not create a branch, edit a repository, mutate DB/Redis,
   start services, or make merchant/API calls.
2. **Offline build/audit:** perform planning, author `mapping.json`, fetch
   read-only specs, build the collection/environment, and audit them. This mode
   still does not modify the target repository, DB, or Redis and makes no live
   merchant calls.
3. **Implementation/local verification:** hand off to the
   `certification-automation` skill after the plan identifies a real code,
   mocker, platform-toggle, seed, or local replay requirement.

When the user says "first perform only planning and inventory", mode 1 is a
hard stop even if the same prompt describes later phases.

## When to use this skill

- A task like "make a Postman collection for these certification testcases"
  with a CSV bearing the contract
  `TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps`.
- A local MCP server (or equivalent catalog) that can return full endpoint
  specs (`get_api_spec`), plus a source repository for semantics checks.

Do NOT use this skill for hand-crafting one-off requests, or when no spec
evidence exists for the target endpoints.

## Inputs to confirm first (ask only if missing and blocking)

1. CSV path.
2. MCP URL (`--mcp-url`, default `http://localhost:8000/mcp`; start the local
   server first: `cd mcp/merchant_mcp && .venv/bin/python src/server/mcp_sse.py`).
3. Output directory (suggest `artifacts/manual/<slug>/`).
4. Source repository path for semantic spot-checks (optional but preferred).
5. Reference Postman collection when it contains known-good prerequisite or
   credential-generation flows.
6. Caller-supplied TSD/NPCI/BCRP/organisation documents when newer protocol
   requirements may not yet exist in the repository or MCP catalog.

Never ask about design; the contract below is the design.

## Workflow (follow in order; do not skip steps)

### 1. Inventory the CSV (seconds)

Input may be XLSX: first convert with
`venv/bin/python3 <skill>/scripts/xlsx_to_csv.py INPUT.xlsx [--sheet "Name"] -o rows.csv`
(sheet names are resolved via workbook relationships — id≠filename). If the
sheet layout differs from the contract
`TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps`, write a
small task-level normalizer that emits the contract CSV inside the output
directory; never overwrite the source. Record the source-to-canonical header
mapping in the plan (for example, `RC-Upi -> RC`). Normalize the sheet's
outcome column honestly — the RC holds the
**wire expectation**, e.g. a payer-declining case is `00-DECLINED`, not
`FAILURE`.

Parse with `csv.DictReader`. Reject missing/duplicate canonical columns and
blank/duplicate testcase IDs. Print and persist: total rows, distinct
`API Name` values with counts and first/last TC ID, distinct `RC` values, and
the normalization performed. Group first by literal `API Name`, then by
**semantic endpoint family**. Every row must belong to exactly one semantic
family, but one API name may split into variants when wire semantics differ.

For planning-only runs, write under the resolved output directory:

- `plan.md`: inputs, source digests, row/header validation, family summary,
  unresolved questions, phase gate, and recommended next command;
- `normalized-testcases.csv`: canonical working copy when normalization was
  needed;
- `grouped-testcases.json`: row IDs grouped by API name and endpoint family;
- `dependency-inventory.json`: one classification per testcase;
- `endpoint-candidates.json`: candidate endpoint IDs, evidence, confidence,
  and rejected candidates;
- `missing-families.json`: missing/ambiguous families and required evidence.

Do not write `mapping.json` during a planning-only run.

### 1.5. Dependency classification (REQUIRED before endpoint research)

For every row, decide **who must act** for the case to pass and persist the
classification in `dependency-inventory.json`. During build, copy non-default
classifications into `mapping.dependencies` (`request` may be omitted there).
The vocabulary is fixed:

| kind | meaning | what the collection does |
|---|---|---|
| `request` | fully expressible in the S2S request body | normal request (default; omit only from `mapping.dependencies`) |
| `env` | needs an operator value (VPA, account id, credBlock) | env placeholder + NOTES entry |
| `bridge` | needs values from an async step (approval, callback UMN) | extraction / paste-bridge env + NOTES |
| `external-simulator` | depends on NPCI/bank-sim/PSP-counterparty test data (INSUFFICIENT_LIMIT, blocked card, delayed/no response) | emit request; flag KIND in dependencies with the reason; never fake the outcome locally |
| `platform-toggle` | needs a behavior flag the platform owns (checksum bypass, feature knobs) | `00 - Test configuration` folder via the toggle API (see escalation) |

Signals: "deliberately", "does not send", "blocked card", "insufficient",
"the Remitter bank declines" → `external-simulator`. "set merchant flag",
"enable feature", "disable verification" → `platform-toggle`.

### Escalation: the code-change track (platform-toggle rows)

If any row classifies as `platform-toggle` and no endpoint/config toggle
exists yet:

1. Stop the collection pipeline for the affected family.
2. Report the requirement table to the user (row, needed behavior, reason).
3. Hand the requirement to `certification-automation`; this skill does not
   edit the platform repository or local data stores.
4. After an explicitly approved implementation is verified, fetch/spec-verify
   the new endpoint, then place its SET and RESET calls
   in the `00 - Test configuration` folder as ordinary families.

Never invent a toggle endpoint, never guess its wire shape — build it,
verify it, then use it. Collections never block silently on
`external-simulator` rows; they ship with the requirement table so operators
know where the case's outcome really comes from.

### 2. Candidate endpoints (minutes)

- Enumerate advertised MCP tools and endpoint IDs. Prefer a full-spec tool
  such as `get_api_spec`; discovery/search results are navigation aids only.
  When the platform repo has a local MCP postgres dump, a read-only catalog
  query is an optional fallback:

  `docker exec mcp-pg psql -U postgres -d mcp_product_context -c "SELECT endpoint_id, method, path FROM endpoint_specs ORDER BY endpoint_id"`

- Map each CSV family to a candidate endpoint id, preferring endpoint specs
  whose description names the internal API (e.g. *"calls NPCI `ReqValAdd`"*).
  Consult supplied TSD/NPCI/BCRP documents for current protocol requirements.
  Use repository grep evidence (`rg -n "<ApiName>" src/`) only to resolve
  ambiguous endpoint semantics; do not bulk-ingest the repository.
- Fetch the FULL spec per candidate (not chunked search results):

  `venv/bin/python3 <skill>/scripts/fetch_specs.py --mcp-url URL --endpoint-id <id> ... --out-dir <out>/specs/`

  A missing endpoint id must be replaced or the family must be flagged as
  unachievable — never proceed with a placeholder endpoint.
- Record zero, one, or multiple candidates per family. Zero is `missing`;
  multiple plausible candidates is `ambiguous`; neither may be silently mapped.

The reference collection is evidence for sequencing, credential generation,
headers, and variable names. It is not wire authority: reconcile it with the
current full spec, source branch, and applicable TSD before copying a request.

### 3. Author mapping.json (the only creative step)

Copy `templates/mapping.bcrp.example.json` as the skeleton and edit:

- One `endpoints` template per distinct wire contract. Build bodies from the
  spec's **Request Schema** section, not from example snippets.
- Work one semantic family at a time and reuse its endpoint template for all
  matching rows; do not synthesize each testcase independently.
- Environment bindings only via `{{ENV_KEY}}`; special tokens
  `__TC_ID__`, `__TEST_DATA_AS_VPA__` (see `reference/mapping-schema.md`).
- Route `when` blocks statically (`description_all` / `description_none` /
  `test_case_id_prefix`).
- Per-row deviations through `row_overrides[TC_ID].body` (inlined after token
  resolution, so overrides always win).
- `environment` triples `[key, default, description]`; secrets = empty string
  default + `SECRET:`-prefixed description.
- Guess-grade decisions (enums, action verbs) go into `notes_markdown` as
  operator checkmarks. Mark them honestly; do not fabricate confidence.
- Model prerequisite flows explicitly and topologically before dependent
  cases. Examples include create-challenge, device binding,
  registration/onboarding, credential-block creation, and response bridges
  needed before balance enquiry, send-money, collect, or mandate calls. Add
  only prerequisites proven by current spec/code/TSD evidence.
- Put platform-toggle SET/RESET calls in `00 - Test configuration`, and list
  every external-simulator requirement in NOTES.

The current builder renders CSV-derived requests and does not yet provide a
generic first-class prerequisite DAG. If independent setup/cleanup requests
cannot be represented without pretending they are testcase rows, mark the
build blocked and extend the mapping schema, builder, and auditor first. Never
hide required API hits only inside opaque pre-request JavaScript.

Read `reference/mapping-schema.md` before writing; it pins the exact contract
and the review-learned authoring rules (e.g. never emit a conditional-empty
`†` field — `existingVpa` incident).

### 4. Build

```
venv/bin/python3 <skill>/scripts/build_collection.py \
  --csv "<csv>" --mapping mapping.json --out-dir <out>/
```

Outputs: `<slug>.collection.json`, `<slug>.environment.json`, `NOTES.md`.

### 5. Audit (MUST pass — do not present the collection before this)

```
venv/bin/python3 <skill>/scripts/audit_collection.py \
  --collection <out>/<slug>.collection.json --environment <out>/<slug>.environment.json \
  --csv "<csv>" --mapping mapping.json --spec-dir <out>/specs/
```

An `AUDIT-FAIL` means fix the mapping and rebuild; never work around it. The
audits encode past review findings: row coverage, duplicate TC IDs,
placeholder coverage (env + computed), mandatory-field conformance vs spec
markdown, `node --check` on every script.

Never edit generated collection or environment JSON manually. Repeat
`mapping.json -> build_collection.py -> audit_collection.py` until the audit
prints `AUDIT-PASS` or a family is honestly blocked.

### 6. Finalize and report

- Fill the operator-TODO section of NOTES.md (values the operator must
  supply, sequencing notes, guess-grade flags).
- Report: item count, folder count, env-key count, audit line, output paths,
  and the honest-unknowns list.
- Write `final-result.json` with source/normalized row counts, semantic-family
  count, rendered testcase count, prerequisite/config request count, blocked
  testcase IDs and reasons, dependency-kind counts, candidate/fetched spec IDs,
  collection/environment/mapping/spec/notes paths, and both gates:
  `static_audit` (`PASS|FAIL`) and `runtime_validation`
  (`PASS|BLOCKED|NOT_RUN`). Set `complete: true` only after `AUDIT-PASS` with
  `rendered + explicitly_blocked == normalized_rows`, no missing/ambiguous
  endpoint for a rendered case, and all required specs present. Because the
  current auditor does not prove prerequisite DAG/reset coverage, secret-empty
  defaults, or complete wire semantics, perform and record those supplemental
  checks before setting `complete: true`. Claim "fully working locally" only
  when the separate runtime gate is `PASS` with no unclassified failures.

## Hard rules

1. **No invented values.** Every literal in a body is either a documented
   constant (`INR`, `"00"`, tokens) or an env binding.
2. **No secrets anywhere.** Templates, notes, scripts — nothing. Fill only in
   the exported environment at execution time.
3. **Full spec docs only.** Chunked `search_docs` results are navigation aids,
   never wire authority; `get_api_spec` output is.
4. **Never ship unfetched endpoints.** A missing/failed spec fetch blocks the
   family — flag it instead of guessing an endpoint.
5. **Audit is the ship gate.** No audit, no delivery claim.
6. Change `mapping.json`, never the generated output, when revising.
7. Keep this skill offline and non-mutating: no target-repository edits, branch
   creation, DB/Redis changes, service startup, or live merchant calls.
8. Use testcase-sheet VPAs and scenario data exactly for request-visible
   values. Never substitute convenient records discovered in a local DB.

## Row-oriented devices that already exist

- RC codes: `0`/`00` → success assertions; combined pairs (`00-00`,
  `00-C08`) → ack-pair comment; anything else → response-contains-code test.
- Item names: `<TC_ID> [expected codes] — <Description, 72 chars>`.
- `_cz_manual` on every item keeps CSV Test Data, steps, and expected RC.
- Config-toggle calls (UAT simulation flags) belong in a
  `00 - Test configuration` folder rendered like any other request — add
  them as normal families, not scripts.

## Time expectations

Inventory < 1 min; candidate research + spec fetch ≈ 1–2 min per endpoint
family (amortized across cases); mapping authoring is the slowest human step;
build + audit < 1 min. A 12-family / 1,000-row sheet ≈ 15–30 min total.
