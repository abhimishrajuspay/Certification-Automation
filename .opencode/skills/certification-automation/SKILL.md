---
name: certification-automation
description: "Run an evidence-gated certification campaign around the certification-postman skill: read-only planning, isolated branch/worktree remediation, scoped local DB/Redis fixtures, local stack/log validation, and audited Postman delivery. Use when a certification suite needs code, mocker, configuration, or local replay work beyond offline collection generation."
---

# Certification Automation skill

A real-world, "we-did-this-for-BCRP" pipeline for Docker-less local validation of a
merchant S2S UPI collection against a running `newton-hs` + `npci-mocking` stack.
Repeatable, minimal SQL, reversible, and framed around pre-flight intake.

## Modes and authorization boundary

Treat these as separate authorizations even when a prompt describes the whole
eventual workflow:

- **Plan/inventory:** read-only. Produce the plan and grouped evidence, then
  stop when the user asks to perform planning first.
- **Offline collection:** may write only run artifacts. No target-repository,
  DB, Redis, service, or merchant API mutation.
- **Implementation:** may create an isolated branch/worktree and change code
  only when the user explicitly asks to proceed with implementation.
- **Local verification:** may start local services, run localhost APIs, and
  make narrowly scoped local DB/Redis changes only when explicitly authorized.
  This never authorizes staging or production calls/data changes.

Absence of approval at a mutation or replay gate means **no**. Persist the
decision and exact scope in `<output-dir>/gate.json`.

## The 9-phase contract

```
[0] INTAKE       -> resolve inputs/digests and exact output directory
[1] PLAN         -> certification-postman read-only inventory; write plan/group/dependency/endpoint reports; STOP when planning-only
[2] WORKTREE     -> from the verified base ref, create an isolated certification branch/worktree (implementation approval)
[3] REMEDIATION  -> implement evidence-backed code/config/mocker changes and focused tests; emit a reversible state plan
[4] STACK        -> start target branch plus required NPCI/mock/callback services and capture separate logs (local-verification approval)
[5] COLLECTION   -> family mapping, secret-empty environment, deterministic build and AUDIT-PASS
[6] FIXTURES     -> apply only approved, run-scoped local DB/Redis fixtures derived from sheet values
[7] REPLAY       -> execute the prerequisite DAG then testcase families; diagnose before changes and stop after three identical failures
[8] FINALIZE     -> rerun build/audit and write final-result.json with separate static/runtime gates
```

Phases 2, 4, 6, and 7 are decision points. A request to plan or build a
collection does not imply permission for them.

## Planning contract

Use the `certification-postman` planning stage and retain its canonical CSV,
header/row validation, API-name and semantic-family groups, one dependency
classification per testcase (`request`, `env`, `bridge`,
`external-simulator`, or `platform-toggle`), advertised MCP tools/candidate
endpoint IDs, and missing/ambiguous family report.

Use repository search only when endpoint semantics need clarification. MCP,
repository, reference-collection, and supplied TSD/NPCI/BCRP/organisation
documents are evidence; the agent remains responsible for the decision.

## Isolated branch/worktree contract

Never switch or edit the caller's existing target checkout, especially when it
is dirty. Resolve the requested base branch locally or as `origin/<name>`, pin
its commit in intake, then create a unique run-named branch in an isolated
worktree. Record base ref/SHA, branch, worktree path, and initial status. Never
push, merge, stash, reset, or commit unless separately requested.

Code changes must cite affected testcase IDs and may cover missing endpoint
support, request fields, response handling, merchant-scoped non-production
configuration, toggle APIs, or focused mocker behavior. Follow current house
patterns and wire evidence; add focused tests and run relevant compile/test/lint
commands before replay.

## Failure triage (every run)

Whenever observed behavior does not match the testcase's expected outcome, do
this BEFORE touching SQL/bodies (an expected decline/error is not a failure):

```
python scripts/diagnose.py <upiRequestId> --db "$CONN" --log /tmp/newton-local.log
```

- prints every tagged error for that requestId from the newton log (all `category`/`level != Info`)
- prints the most-meaningful `newton-*` redis keys (merchant / config / customer / txn)
- runs the LC-scoped probes from `reference/recipes.md` (customer chain, account, linked, merchant config, fallback config)
- `--clear` additionally invalidates merchant/config/customer redis keys

Use `diagnose.py` only after checking its configured merchant/key scope against
the current plan. Its `--clear` mode is a Redis mutation and requires the
separate fixture-mutation approval.

The skill's iron law: 3 consecutive same-error attempts == STOP and read the
recipe's ladder again, never guess SQL randomly.

Each phase persists artifacts under the caller's exact resolved output
directory so a human can replay it. Do not store credential-bearing DB URLs;
persist a redacted connection alias/hash. Every proposed DB/Redis mutation must
have a scoped target list, pre-state evidence, rollback, and a separate apply
approval. The existing helper scripts do not by themselves prove this safety.

## Intake answers (ask once, at phase 0)

Collect intake fields once. Request each later mutation/replay gate separately.
Defaults when the answers are known:

| Field                       | Example (BCRP Comfort v1)                                       |
|-----------------------------|-----------------------------------------------------------------|
| `testcase_csv`              | `Test-case-BCRP-Comfort-v1 - Sheet1 copy.csv`                   |
| `reference_collection`      | `Newton.postman_collection.json` (has working bodies+prereqs)   |
| `dummy_collection`          | Dummy postman collection carrying **cred/credential-block (CL) generation logic** — the skill extracts `pm.sendRequest(http://localhost:<port>/createCredBlock, …)` headers+body from its prerequest scripts so drafted bodies share the same encryption flow |
| `credblock_helper`          | `http://localhost:4590/createCredBlock` (or equivalent helper discovered in the dummy collection) |
| `repo_path`                 | `/Users/…/repos/newton-hs` (validated wire types live here)     |
| `mocker`                    | `npci-mocking/` path + `npm run dev` + port 8089                |
| `callback_base`             | `http://localhost:8012` (must match the newton's bind)          |
| `run_after_build`           | yes/no — whether the later approved replay phase proceeds (default: **no**) |
| `db`                        | host/user/database + table prefix whitelist (LC- only)          |
| `redis`                     | `redis-cli` path; expect `newton-*` key namespace               |
| `merchant_id`               | `TESTMERCHANT` / id `A117f0ea…b79`                              |
| `merchant_api_key`          | NOT stored — only referenced via env alias                      |
| `checksum_bypass`           | `localcert` (must match MerchantConfigurations.checksumBypassId) |
| `credentials`               | merchantCustomerId, customerMobile, device profile              |
| `base_branch`               | caller-selected starting ref, e.g. `temp/peru-staging`           |
| `tsd_sources`                | supplied current TSD/NPCI/BCRP/org documents                     |
| `output_dir`                 | exact caller-selected absolute or workspace-relative run path    |

`scripts/intake.py` is a non-interactive argparse helper, not a complete generic
inventory validator. Supply inputs explicitly and write `intake.json` under the
resolved output directory. During planning-only mode, availability probes are
read-only and DB/Redis/mocker availability is not required to finish inventory.

## Why the dummy/ref collection matters
It is behavioral evidence for local CL generation and prerequisite sequencing:
1. The `createCredBlock` helper URL/port (e.g. `http://localhost:4590/createCredBlock`;
   BCRP's `Newton.postman_collection.json` declares 3: two remote + localhost).
2. The exact JSON body (token,type=cred,salt fields, amount/txnId/vpas/appId/deviceId).
3. Where the response token+ki land (variable names like `npciToken-CredBlockMPIN`,
   `credBlockToken`, `ki`).
Do not assume `replay_local.py` parses a supplied reference collection: today it
does not. Extract required behavior during mapping and represent challenge/
credential steps explicitly. Never fall back to fabricated credentials when
the current testcase/spec requires a real challenge flow; block the case.

## Replay-ladder contracts (lean, current)

- `scripts/replay_local.py` derives a validator-safe alphanumeric `REQUEST_ID`
  when the environment does not prescribe one — prevents Mongo-type collisions
  on the test rpc without chain-mode semantics; pass `--set REQUEST_ID=...` to
  force a single value (chain flows like ChkTxn→Pay need that).
- Folder execution order follows the current plan's prerequisite DAG, never
  CSV order alone. For example, a ChkTxn flow may need a prior Pay/pushToVpa
  response's real `merchantRequestId`/`gatewayReferenceId`; retain extracted
  values in the run evidence and bridge them explicitly.
- `diagnose.py` reads the log at the moment of failure only; rotate the live log
  into `out/<run-id>/logs/` before any restart with:
  `cp /tmp/newton-local.log out/<run-id>/logs/$(date +%H%M%S).log`.

## What the skill does NOT do

- Never commits or persists secrets: apiKeys/apiKeyHash and credential-bearing
  connection strings stay environment-only.
- Never wipes DB tables or broadly deletes Redis keys. Schema migrations,
  shared configuration updates, and existing merchant-row updates are code/
  migration changes requiring their own review, not ordinary test fixtures.
- Never pushes git branches; never merges anything.
- Never fabricates NPCI responses; every behavior is observed from live calls.
- Never calls staging or production merchant/NPCI endpoints; replay is against
  caller-approved local services only.
- Never reuses convenient VPAs, mobiles, accounts, or scenario values found in
  the DB. Request-visible data comes from the testcase sheet/reference inputs.
  Internal IDs, when unavoidable, are deterministic, run-scoped, and recorded.
- Never edits generated Postman JSON. Correct `mapping.json`, rebuild, reaudit.

The current `run_all.sh`, `seed_ladder.py`, `stack_bringup.sh`, and
`replay_local.py --auto-*` paths contain BCRP/machine-specific assumptions or
hidden mutations. Treat them as historical helpers, not generic certification
commands. Do not use them for a new suite until their hardcoded identities,
mutation scope, exit status, and rollback behavior have been audited against
the current plan. A printed `DOWN`, unresolved variable, transport failure, or
non-matching response must produce a non-passing runtime result even if a
legacy helper exits zero.

## Collection and prerequisite contract

Build `mapping.json` family-by-family, not testcase-by-testcase. Use the
reference collection to discover known-good orchestration and credential logic,
then verify each current wire shape against full specs, source, and applicable
TSDs. Include evidence-backed prerequisite calls in topological order,
including create-challenge, device binding, registration/onboarding,
credential-block creation, and response bridges required before balance
enquiry, send-money, collect, mandate, or related flows.

List every `platform-toggle` and `external-simulator` requirement explicitly.
Configuration requests need safe setup and cleanup/reset behavior. Secrets stay
empty in the generated environment template. Run `build_collection.py` and
`audit_collection.py`; on `AUDIT-FAIL`, revise `mapping.json` (or extend the
generic generator/auditor if the schema lacks a required construct), then
rebuild. Never patch generated JSON.

## Local evidence and mutation contract

Before a local fixture mutation, write the proposed SQL/Redis operation,
affected testcase IDs, current state, exact keys/rows, and rollback into the
output directory. Apply only to the approved local instance and only for
sheet-derived scenarios. Capture timestamped logs separately for newton and
every NPCI/mocker/callback/credential-helper service; correlate them by testcase
and request ID and retain sanitized excerpts or hashes.

For each family, retain request/response status, expected versus actual
envelope/gateway/callback/state outcome, service-log citations, prerequisite
outputs, and the code/config/fixture revision. Inspect code and all relevant
logs before attributing failure. External-simulator behavior remains an
external requirement; never mask it by changing the platform response.

Write `final-result.json` with source/normalized/rendered counts, prerequisite
and configuration request counts, dependency/support/blocked classifications,
branch/worktree and code-verification results, local replay coverage, all output
paths, and two independent gates: `static_audit` (`PASS|FAIL`) and
`runtime_validation` (`PASS|BLOCKED|NOT_RUN`). `complete: true` requires
`AUDIT-PASS`; a "fully working locally" claim additionally requires runtime
`PASS` with no unclassified failures.

## Phase 6 — config toggle plan (skeleton)

Required by cases that need dynamic behaviour (e.g. `blockDirectPay`, partial
UPDA fallback, checksum enforcement). The following is only an architectural
example, not an endpoint to assume exists. Derive the real route, auth, scope,
storage, TTL, and reset semantics from the target branch:

```
PUT /api/x2/admin/s2s/config-toggles      (Vault + internal-api-key header)
{ "key": "blockDirectPay", "value": "true", "ttlSec": 900 }
```

Newton change: `Services/ConfigOverrides.store` + `mergeInto ConfigurationData`
in the existing lattice. Evidence list in the plan doc must name the exact
testcase id that requires the toggle. Nothing ships until THIS plan's test
case list is signed off.

## The replay ladder (what "validating" actually means)

A row counts as PASS only when observed HTTP, envelope, gateway code, callback,
log, and state outcomes match that row's normalized expectation. Expected
declines/errors may therefore pass; HTTP 200/SUCCESS is not universal. A folder
is re-run only after evidence-backed changes and until every executable row
passes or is honestly blocked. Fixes land between runs as reviewed code,
fixture, config, mocker, or mapping changes—never as hidden replay mutations.

## Required reading before ANY SQL against newton's DB

`reference/recipes.md` is historical BCRP evidence, not a generic seed script.
Use it to recognize failure categories, but never copy its identifiers, VPAs,
SQL, Redis keys, or expected codes unless the current sheet, schema, and plan
independently support them. Generate a current dry-run mutation/rollback plan
and obtain approval before applying any SQL or Redis change.
