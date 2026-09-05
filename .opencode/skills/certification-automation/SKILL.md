---
name: certification-automation
description: End-to-end local certification-workflow for a newton-hs API surface: intake (reference Postman collection, repo path, NPCI mocker, DB+redis), branch + DB seed plan, local stack bring-up, collection build via the certification-postman skill, and live replay ladder until every row is envelope-validated. Use when asked to "certify" a CSV testcase sheet against a newton-family backend, or in any flow that starts with 'make a new skill that does intake/repository/mocker/db+redis and runs everything we had to do for BCRP/MDC'.
---

# Certification Automation skill

A real-world, "we-did-this-for-BCRP" pipeline for Docker-less local validation of a
merchant S2S UPI collection against a running `newton-hs` + `npci-mocking` stack.
Repeatable, minimal SQL, reversible, and framed around pre-flight intake.

## The 7-phase contract

```
[0] INTAKE       -> resolve every input, write out/intake.json
[1] PLAN         -> parse the CSV, diff vs code ground truth, write plan.md
[2] BRANCH+SEED  -> create git branch, emit seeds.sql (LC- prefixed, narrow)
[3] STACK-BRINGUP-> newton + mocker + callback conduct + env overrides
[4] COLLECTION   -> certification-postman skill: build_collection.py + audit
[5] REPLAY-GATE  -> ASK: "run now against the local stack to verify changes +
                   collection?" — driver replay_local.py, LC- environment set
[6] TOGGLE API   -> propose "config toggle" HTTP surface IF required by cases
```

Phase 5 is a DECISION point by design: the operator should know BEFORE any
live call is made that the flow now depends on a stack check, DB seeds, and
collection behaviour. Default: YES, but every gate answer lands in
`out/<run-id>/gate.json` so the run is auditable.

## Failure triage (every run)

Whenever a replay folder produces ANY non-SUCCESS envelope, do this BEFORE
touching SQL/bodies:

```
python scripts/diagnose.py <upiRequestId> --db "$CONN" --log /tmp/newton-local.log
```

- prints every tagged error for that requestId from the newton log (all `category`/`level != Info`)
- prints the most-meaningful `newton-*` redis keys (merchant / config / customer / txn)
- runs the LC-scoped probes from `reference/recipes.md` (customer chain, account, linked, merchant config, fallback config)
- `--clear` additionally invalidates merchant/config/customer redis keys

The skill's iron law: 3 consecutive same-error attempts == STOP and read the
recipe's ladder again, never guess SQL randomly.

Each phase persists artifacts under `out/<run-id>/` so a human can replay them
exactly. Every DB write is inside a small `BEGIN/COMMIT` and uses the `LC-`
prefix — nothing else is touched.

## Intake answers (ask once, at phase 0)

Ask the operator ONCE, then never again. Defaults when the answers are known:

| Field                       | Example (BCRP Comfort v1)                                       |
|-----------------------------|-----------------------------------------------------------------|
| `testcase_csv`              | `Test-case-BCRP-Comfort-v1 - Sheet1 copy.csv`                   |
| `reference_collection`      | `Newton.postman_collection.json` (has working bodies+prereqs)   |
| `dummy_collection`          | Dummy postman collection carrying **cred/credential-block (CL) generation logic** — the skill extracts `pm.sendRequest(http://localhost:<port>/createCredBlock, …)` headers+body from its prerequest scripts so drafted bodies share the same encryption flow |
| `credblock_helper`          | `http://localhost:4590/createCredBlock` (or equivalent helper discovered in the dummy collection) |
| `repo_path`                 | `/Users/…/repos/newton-hs` (validated wire types live here)     |
| `mocker`                    | `npci-mocking/` path + `npm run dev` + port 8089                |
| `callback_base`             | `http://localhost:8012` (must match the newton's bind)          |
| `run_after_build`           | yes/no — whether [5] automatically proceeds to replay after collection build (default: **yes**, always asked) |
| `db`                        | host/user/database + table prefix whitelist (LC- only)          |
| `redis`                     | `redis-cli` path; expect `newton-*` key namespace               |
| `merchant_id`               | `TESTMERCHANT` / id `A117f0ea…b79`                              |
| `merchant_api_key`          | NOT stored — only referenced via env alias                      |
| `checksum_bypass`           | `localcert` (must match MerchantConfigurations.checksumBypassId) |
| `credentials`               | merchantCustomerId, customerMobile, device profile              |

`scripts/intake.py` asks these, validates reachability of repo/db/redis/mocker,
and writes `out/<run-id>/intake.json`.

## Why the dummy/ref collection matters
It is the ONLY trusted source of local CL generation logic — the replay reads
its prerequest scripts to learn:
1. The `createCredBlock` helper URL/port (e.g. `http://localhost:4590/createCredBlock`;
   BCRP's `Newton.postman_collection.json` declares 3: two remote + localhost).
2. The exact JSON body (token,type=cred,salt fields, amount/txnId/vpas/appId/deviceId).
3. Where the response token+ki land (variable names like `npciToken-CredBlockMPIN`,
   `credBlockToken`, `ki`).
The replay driver substitutes STAGED mock creds by the actual minted token when
the helper is reachable; otherwise falls back to the validated Juspay recipe
(`2.0|DUMMYBASE64…` shape documented in `reference/recipes.md`).

## Replay-ladder contracts (lean, current)

- `scripts/replay_local.py` derives `REQUEST_ID` per item as `LC{TC_ID}_{epoch_ms}`
  when the environment does not prescribe one — prevents Mongo-type collisions
  on the test rpc without chain-mode semantics; pass `--set REQUEST_ID=...` to
  force a single value (chain flows like ChkTxn→Pay need that).
- Folder execution order follows the CSV, with ChkTxn AFTER Pay/pushToVpa —
  its lookups require real `merchantRequestId`/`gatewayReferenceId` values that
  only appear after 05a/05b has run. The evidence ledger
  `out/<run-id>/replay/<folder>.jsonl` carries `gatewayTransactionId` per row;
  ChkTxn replays pull it back with `--set MERCHANT_REQUEST_ID=<id>`.
- `diagnose.py` reads the log at the moment of failure only; rotate the live log
  into `out/<run-id>/logs/` before any restart with:
  `cp /tmp/newton-local.log out/<run-id>/logs/$(date +%H%M%S).log`.

## What the skill does NOT do

- Never commits secrets: apiKeys/apiKeyHash stay in local `.env.local` only;
  `out/<run-id>/intake.json` may contain a DB conn-string — make sure
  `out/**` is gitignored BEFORE first run.
- Never wipes DB tables; only LC-prefixed rows + `ALTER TYPE ADD VALUE`.
- Never pushes git branches; never merges anything.
- Never fabricates NPCI responses; every behavior is observed from live calls.

## Phase 6 — config toggle plan (skeleton)

Required by cases that need dynamic behaviour (e.g. `blockDirectPay`, partial
UPDA fallback, checksum enforcement). Tile: ONE route, JSON body + dedicated
Redis override that configFetch picks up within 5 seconds:

```
PUT /api/x2/admin/s2s/config-toggles      (Vault + internal-api-key header)
{ "key": "blockDirectPay", "value": "true", "ttlSec": 900 }
```

Newton change: `Services/ConfigOverrides.store` + `mergeInto ConfigurationData`
in the existing lattice. Evidence list in the plan doc must name the exact
testcase id that requires the toggle. Nothing ships until THIS plan's test
case list is signed off.

## The replay ladder (what "validating" actually means)

A row counts as PASS only when its merchant API returns a 200 SUCCESS envelope
and `payload.gatewayResponseCode` is captured. The replay driver
`scripts/replay_local.py`: a folder is re-run until either every row passes or
the error reaches an un-patchable category; the folder result list is the
report. Fixes must land BETWEEN runs as seeds/config/body-tuning — never inside
the replay script itself.

## Required reading before ANY SQL against newton's DB

`reference/recipes.md` — the exact DB-error → fix ladder observed while
validating BCRP-Comfort-v1, in order, with the proven SQL. Derived from
evidence, NOT from guesswork. Each step also lists the RPC error shape (e.g.
`INVALID_DATA/Vpa not found`) so future operators can map new errors to existing
remedies before inventing new ones.
