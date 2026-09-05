# mapping.json contract (certification-postman)

`mapping.json` is the ONLY per-task creative input. Everything else
(`build_collection.py`, `audit_collection.py`) consumes it mechanically.

```jsonc
{
  "name": "BCRP Comfort Certification v1",          // collection name
  "description": "...",                              // collection description

  "endpoints": {                                     // keyed wire templates
    "VALIDITY": {
      "method": "POST",
      "path": "/api/{{API_VERSION}}/merchants/vpas/validity",
      "spec": "newton.s2s.post.merchants.vpas.validity", // <out>/specs/<id>.md fetched via get_api_spec — REQUIRED for the mandatory-field audit
      "body": { "...": "..." }                       // null for GET
    }
  },

  "families": [                                      // CSV "API Name" -> endpoint routing (first match wins)
    {"api_names": ["ReqValAdd", "ReqGetAdd"], "folder": "01 - ...", "endpoint": "VALIDITY"},
    {"api_names": ["ReqSetCre"], "routes": [
      {"when": {"description_all": ["chang"], "description_none": ["reset"]}, "folder": "03 - Change ...", "endpoint": "CHANGE_MPIN"},
      {"folder": "03 - Set/Reset ...", "endpoint": "SET_MPIN"}   // route with no "when" is the family default
    ]},
    {"api_names": ["ReqPay"], "routes": [
      {"when": {"test_case_id_prefix": ["TC_PR"]}, "folder": "05b - Push", "endpoint": "PUSH_TO_VPA"},
      {"folder": "05a - Collect", "endpoint": "COLLECT"}
    ]}
  ],

  "row_overrides": {                                 // per-row body tweaks; applied AFTER route body, always wins
    "MT_02": {"body": {"customerVpa": "{{MOBILE_NUMBER}}@mapper.npci"}},
    "PE_04": {"body": {"requestType": "DECLINE"}}
  },

  "computed_vars": [                                 // JS lines appended to the collection prerequest (fresh values per run)
    "pm.environment.set(\"VSTART\", fmtD(new Date()));"
  ],

  "dependencies": [                                  // rendered as a requirement table in NOTES.md
    {"tc": "MA_CC_RE_8", "kind": "external-simulator", "reason": "Remitter bank sim must withhold RespMandate (timeout)"}
  ],                                                // kinds: env | bridge | external-simulator | platform-toggle

  "environment": [                                   // [key, default, description] — secrets stay empty strings
    ["MERCHANT_API_KEY", "", "SECRET: HMAC key — never commit"]
  ],

  "notes_markdown": "## Operator notes\n\n- ...\n"   // appended verbatim into <out>/NOTES.md
}
```

## Body-template value rules

| Value | Meaning |
|---|---|
| `"{{ENV_KEY}}"` | environment variable binding; emitted literally, must exist in `environment` (audit enforces) |
| `"__TC_ID__"` | current row's TC ID, inlined at build time |
| `"__TEST_DATA_AS_VPA__"` | row's Test Data inlined when it matches `local-part@handle`; otherwise becomes `{{CUSTOMER_VPA}}` |
| any other string | emitted literally (examples: `"INR"`, `"00"`, `"false"`) |
| objects/arrays | token substitution recurses inside |

## Extended taxonomy

- Route `when`: `description_all` ANDs terms; `description_none` rejects on
  terms; `description_any` requires ONE of the terms; `test_case_id_prefix`
  any-of TC prefixes.
- A route may carry `"body"` overrides — merged after the endpoint template,
  before `row_overrides`; value `null` **deletes** a key (use to drop
  variant-only fields, e.g. `pauseStart`/`pauseEnd` on UNPAUSE rows).
- Endpoint extras:
  - `"headers": {"x-api-version": "2"}` — fixed per-request headers.
  - `"assertion": {"success_body_regex": "...", "failure_body_regex": "..."}` —
    replaces the generic success/failure test for rows of that endpoint
    (precise gateway-level patterns like `"gatewayResponseCode":"0[01]"`).
    JSON-escape backslashes: `\\s`, `\\d`.
  - `"extract": {"ENV_KEY": "json.dotted.path"}` — appends a capture test that
    persists a response field into the environment so later folders can use it
    (create → lifecycle chains).
- **Folders are emitted sorted by name** — always zero-pad prefixes
  ("01..".."11") so runtime order is pinned independent of CSV row order
  (mandate sheets list execution rows before creation rows!).
- Sheet normalization: the pipeline consumes the contract header
  `TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps`. For
  foreign layouts write a small normalizer first; map the sheet's outcome
  column honestly — e.g. `STATUS=Failure` → `RC=FAILURE` when the wire call
  itself fails, but `00-DECLINED` when the expected outcome is a *successful
  decline* (`MA_CC_PE_4` lesson).

## Authoring rules (learned from the BCRP review pass)

1. **Never emit a conditional field with an env placeholder as its only guard.**
   If a spec marks a field `†` conditional AND "must be non-empty if supplied",
   include it only for rows where the condition truly holds (use
   `row_overrides` or a dedicated route); empty env would fail server-side
   validation.
2. Read the spec's **Request Schema** section — not only its Examples. The
   audit compares your template against every `*`-required field
   (`- **name***` lines, dot paths flattened).
3. Mark guess-grade values (enum picks, action verbs) in `notes_markdown`
   rather than hiding them.
4. Keep secrets out of templates and descriptions; secrets only as empty env
   defaults with a `SECRET:`-prefixed description.
