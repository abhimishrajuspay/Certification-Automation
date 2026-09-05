---
name: certification-postman
description: Use when generating a Postman certification collection from a testcase CSV (e.g. "Test-case-... Sheet1.csv") mapped onto an API platform whose endpoints are introspectable via a local MCP get_api_spec tool or equivalent spec docs. Parses the CSV, maps API families to wire endpoints, builds Collection v2.1 + secret-empty environment + NOTES, and audits everything. Not for one-off ad-hoc API requests.
---

# Certification Postman

Deterministic pipeline: **CSV → endpoint mapping (mapping.json) → build → audit → ship.**
The only creative input is `mapping.json`. Scripts do everything else. Target
runtime: minutes even for 1,000+ rows, because evidence amortizes per endpoint
family, not per case.

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

Never ask about design; the contract below is the design.

## Workflow (follow in order; do not skip steps)

### 1. Inventory the CSV (seconds)

Input may be XLSX: first convert with
`venv/bin/python3 <skill>/scripts/xlsx_to_csv.py INPUT.xlsx [--sheet "Name"] -o rows.csv`
(sheet names are resolved via workbook relationships — id≠filename). If the
sheet layout differs from the contract
`TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps`, write a
small task-level normalizer that emits the contract CSV (keep it next to the
output dir). Normalize the sheet's outcome column honestly — the RC holds the
**wire expectation**, e.g. a payer-declining case is `00-DECLINED`, not
`FAILURE`.

Parse with `csv.DictReader`. Print: total rows, distinct `API Name` values
with counts and first/last TC ID, distinct `RC` values. Every distinct
`API Name` must resolve to exactly one endpoint family — list any unmapped
families immediately.

### 1.5. Dependency classification (REQUIRED before endpoint research)

For every row, decide **who must act** for the case to pass and write it
into `mapping.dependencies` (kind vocabulary is fixed):

| kind | meaning | what the collection does |
|---|---|---|
| `request` | fully expressible in the S2S request body | normal request (default — omit) |
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
3. With user confirmation, implement the toggle in the platform repo on a new
   branch, **following the house pattern**: honor-only-when
   `EnvKey /= "production"`, merchant/scoped storage (configuration row or
   `/test/*` route family like `x-bypass-merchant-checksum`/`test/merchantAuth`).
4. Fetch/spec-verify the new endpoint, then place its SET (and RESET) calls
   in the `00 - Test configuration` folder as ordinary families.

Never invent a toggle endpoint, never guess its wire shape — build it,
verify it, then use it. Collections never block silently on
`external-simulator` rows; they ship with the requirement table so operators
know where the case's outcome really comes from.

### 2. Candidate endpoints (minutes)

- Enumerate available endpoint ids. When the platform repo has a local MCP
  postgres dump, query it directly (fastest, complete):

  `docker exec mcp-pg psql -U postgres -d mcp_product_context -c "SELECT endpoint_id, method, path FROM endpoint_specs ORDER BY endpoint_id"`

- Map each CSV family to a candidate endpoint id, preferring endpoint specs
  whose description names the internal API (e.g. *"calls NPCI `ReqValAdd`"*).
  Use repository grep evidence (`rg -n "<ApiName>" src/`) to disambiguate.
- Fetch the FULL spec per candidate (not chunked search results):

  `venv/bin/python3 <skill>/scripts/fetch_specs.py --mcp-url URL --endpoint-id <id> ... --out-dir <out>/specs/`

  A missing endpoint id must be replaced or the family must be flagged as
  unachievable — never proceed with a placeholder endpoint.

### 3. Author mapping.json (the only creative step)

Copy `templates/mapping.bcrp.example.json` as the skeleton and edit:

- One `endpoints` template per distinct wire contract. Build bodies from the
  spec's **Request Schema** section, not from example snippets.
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

### 6. Finalize and report

- Fill the operator-TODO section of NOTES.md (values the operator must
  supply, sequencing notes, guess-grade flags).
- Report: item count, folder count, env-key count, audit line, output paths,
  and the honest-unknowns list.

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
