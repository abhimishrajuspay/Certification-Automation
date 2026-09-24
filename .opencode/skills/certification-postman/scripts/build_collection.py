#!/usr/bin/env python3
"""Deterministic certification-collection builder.

Usage (from the Certification-Automation repo root):

  venv/bin/python3 .opencode/skills/certification-postman/scripts/build_collection.py \
    --csv "path/to/testcases.csv" --mapping mapping.json --out-dir out/

Inputs:
  --csv       testcase CSV with the exact header contract
              "TC ID,API Type,API Name,Test Data,Version,RC,Description,Steps"
  --mapping   JSON file describing endpoints, family routing, row overrides,
              environment keys (see reference/mapping-schema.md)
  --out-dir   output directory (created)

Outputs in --out-dir:
  <name>.collection.json   Postman Collection v2.1
  <name>.environment.json  secret-empty environment template
  NOTES.md                 prose notes: mapping evidence, assertions, TODOs

Body-template value tokens understood by this builder:
  "{{ENV_KEY}}"              environment variable binding (stays literal)
  "__TC_ID__"                the current row's TC ID (inlined at build time)
  "__TEST_DATA_AS_VPA__"     the row's Test Data inlined when it matches a
                             VPA shape (local-part@handle), else {{CUSTOMER_VPA}}

Overrides in mapping "row_overrides"[TC_ID]["body"] replace template keys
added AFTER token resolution, so they always win over defaults and tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

VPA_SHAPE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\Z")

COLLECTION_PREREQUEST = """\
// Certification collection — plain JSON + HMAC envelope + NPCI credBlock generation.
// Recipe (newton-hs: src/Newton/Utils/Crypto.hs verifyRequestSignature +
// App/Middlewares/Authentication/MerchantSignatureVerificationV2.hs):
//   x-merchant-signature = lowercase-hex HMAC-SHA256(
//     key = MERCHANT_API_KEY,
//     msg = merchantId + channelId + subMerchantId + subChannelId + ts + rawBody)
// NPCI credBlock: obtained from the local credBlock service
//   (http://localhost:4590/createChallenge + /createCredBlock)
//   which mirrors the reference collection's pre-request pattern.
const ts = Date.now().toString();
pm.variables.set("IAT", ts);
const tcMatch = pm.info.requestName.match(/^([A-Za-z_0-9-]+)/);
const cleanTc = (tcMatch ? tcMatch[1] : "REQ").replace(/[^A-Za-z0-9]/g, "");
pm.variables.set("REQUEST_ID", ("CERT" + cleanTc + ts).substring(0, 35));
const resolve = (value) => value.replace(/\\{\\{([^}]+)\\}\\}/g, (m, key) => {
  const replacement = pm.variables.get(key.trim());
  return replacement === undefined ? m : replacement;
});
let rawBody = "";
try {
  const body = pm.request.body;
  if (body && body.mode === "raw" && typeof body.raw === "string") {
    rawBody = resolve(body.raw);
  }
} catch (_) { rawBody = ""; }
const mid = pm.variables.get("MERCHANT_ID") || "";
const chId = pm.variables.get("MERCHANT_CHANNEL_ID") || "";
const subMid = pm.variables.get("SUB_MERCHANT_ID") || "";
const subChId = pm.variables.get("SUB_MERCHANT_CHANNEL_ID") || "";
const apiKey = pm.variables.get("MERCHANT_API_KEY") || "";
if (!mid || !chId || !apiKey) {
  console.warn("CERT: merchant credentials not set; platform will reject the call.");
}
const signature = CryptoJS.HmacSHA256(mid + chId + subMid + subChId + ts + rawBody, apiKey)
  .toString(CryptoJS.enc.Hex);
pm.request.headers.upsert({ key: "x-merchant-id", value: mid });
pm.request.headers.upsert({ key: "x-merchant-channel-id", value: chId });
if (subMid) pm.request.headers.upsert({ key: "x-sub-merchant-id", value: subMid });
if (subChId) pm.request.headers.upsert({ key: "x-sub-merchant-channel-id", value: subChId });
pm.request.headers.upsert({ key: "x-timestamp", value: ts });
pm.request.headers.upsert({ key: "x-merchant-signature", value: signature });
pm.request.headers.upsert({ key: "Content-Type", value: "application/json" });

// ── NPCI credBlock generation (only for credBlock-dependent requests) ──
// Pattern from the reference Newton collection: get an npciToken via createChallenge,
// then create credBlocks via createCredBlock. The raw responses go into
// data.encryptedBase64String; data.ki = "20150822" is the NPCI key identifier.
//
// Per-endpoint credBlock types (matching the Newton request wire contract):
//   balance    → mpincred (type=PIN, subType=MPIN)
//   setMpin    → otpcred (type=OTP) + atmpincred (type=PIN, subType=ATMPIN) + mpincred (type=PIN, subType=MPIN)
//   changeMpin → mpincred (type=PIN, subType=MPIN, current PIN) + newcred (type=PIN, subType=NMPIN, new PIN)
const credServiceUrl = pm.variables.get("CRED_SERVICE_URL") || "http://localhost:4590";
const deviceRawFp = pm.variables.get("DEVICE_RAW_FINGERPRINT") || "";
const packageName = pm.variables.get("PACKAGE_NAME") || pm.variables.get("APP_PACKAGE_NAME") || "";
const mobileNumber = pm.variables.get("MOBILE_NUMBER") || "";
const customerVpa = pm.variables.get("CUSTOMER_VPA") || "";
const payeeVpa = pm.variables.get("MERCHANT_VPA") || "";
const requestId = pm.variables.get("REQUEST_ID") || "";
const amount = pm.variables.get("AMOUNT") || "1.00";
const mpin = pm.variables.get("UPI_MPIN") || "123456";
const newMpin = pm.variables.get("NEW_MPIN") || "654321";
const atmPin = pm.variables.get("ATM_PIN") || "123456";
const otpValue = pm.variables.get("OTP_VALUE") || "123456";

// Only generate credBlock if the request body needs it
const needsCredBlock = rawBody.includes("{{UPI_CRED_BLOCK}}") || rawBody.includes("{{MPIN_CRED_ENCRYPTED_BASE64}}");

// Detect the endpoint from the request URL to determine which credBlock types are needed
const requestUrl = pm.request.url.toString ? pm.request.url.toString() : String(pm.request.url);
const credTypes = [];
if (requestUrl.includes("setMpin")) {
  // setMpin: OTP + ATMPIN + MPIN (debit card + OTP + new MPIN flow)
  credTypes.push({ blockKey: "otpcred",     type: "OTP",  subType: "SMS",   cred: otpValue });
  credTypes.push({ blockKey: "atmpincred",  type: "PIN", subType: "ATMPIN", cred: atmPin });
  credTypes.push({ blockKey: "mpincred",    type: "PIN", subType: "MPIN",  cred: mpin });
} else if (requestUrl.includes("changeMpin")) {
  // changeMpin: current MPIN + new MPIN
  credTypes.push({ blockKey: "mpincred", type: "PIN", subType: "MPIN", cred: mpin });
  credTypes.push({ blockKey: "newcred",  type: "PIN", subType: "NMPIN", cred: newMpin });
} else {
  // balance (and any other credBlock endpoint): MPIN only
  credTypes.push({ blockKey: "mpincred", type: "PIN", subType: "MPIN", cred: mpin });
}

if (needsCredBlock && deviceRawFp && packageName) {
  // Step 1: get an NPCI token via createChallenge
  const challengeRequest = {
    url: credServiceUrl + "/createChallenge",
    method: "POST",
    header: { "Content-Type": "application/json" },
    body: {
      mode: "raw",
      raw: JSON.stringify({
        deviceId: deviceRawFp,
        appId: packageName,
        type: "initial"
      })
    }
  };

  pm.sendRequest(challengeRequest, (err, res) => {
    if (err) {
      console.warn("CERT: credBlock challenge failed:", err.message);
      pm.variables.set("UPI_CRED_BLOCK", "{}");
      return;
    }
    const challengeResponse = res.json();
    const npciToken = challengeResponse["token"];

    // Step 2: generate a credBlock for each required type, then assemble
    const credBlocks = {};
    let completed = 0;

    credTypes.forEach((ct) => {
      const credBlockRequest = {
        url: credServiceUrl + "/createCredBlock",
        method: "POST",
        header: { "Content-Type": "application/json" },
        body: {
          mode: "raw",
          raw: JSON.stringify({
            token: npciToken,
            type: ct.type,
            cred: ct.cred,
            salt: {
              txnAmount: amount,
              txnId: requestId,
              payerAddr: customerVpa,
              payeeAddr: payeeVpa,
              appId: packageName,
              mobileNumber: mobileNumber,
              deviceId: deviceRawFp
            }
          })
        }
      };

      pm.sendRequest(credBlockRequest, (err2, res2) => {
        completed++;
        if (err2) {
          console.warn("CERT: credBlock creation failed for", ct.blockKey + ":", err2.message);
        } else {
          const credResponse = res2.json().toString();
          // The response is a quoted string like "NPCI,20150822,2.0|<encrypted>"
          const credValue = credResponse.replace(/^"|"$/g, "");
          credBlocks[ct.blockKey] = {
            type: ct.type,
            subType: ct.subType,
            data: {
              type: "",
              skey: "",
              pid: "",
              ki: "20150822",
              hmac: "",
              encryptedBase64String: credValue,
              code: "NPCI"
            }
          };
        }

        // When all types are done, set the assembled credBlock variables.
        // CONTRACT: the template embeds credBlock inside JSON-string quotes
        // ("credBlock": "{{UPI_CRED_BLOCK}}"), so the variable must carry
        // BACKSLASH-ESCAPED JSON (no outer quotes) for the wire body to stay
        // valid JSON. Operator-pasted values must follow the same contract.
        if (completed === credTypes.length) {
          const credBlockJson = JSON.stringify(credBlocks);
          pm.variables.set("UPI_CRED_BLOCK", JSON.stringify(credBlockJson).slice(1, -1));
          const mpinCred = credBlocks.mpincred || credBlocks.newcred;
          if (mpinCred) {
            pm.variables.set("MPIN_CRED_ENCRYPTED_BASE64", mpinCred.data.encryptedBase64String);
            pm.variables.set("MPIN_CRED_KEY_ID", "20150822");
          }
          console.log("CERT: credBlock generated for", cleanTc, "- types:", Object.keys(credBlocks).join(","));
        }
      });
    });
  });
}
"""


def expected_codes(rc: str) -> list[str]:
    parts = [part.strip() for part in rc.split("-") if part.strip()]
    return parts or ["00"]


def success_expected(rc: str) -> bool:
    return all(part in {"0", "00"} for part in expected_codes(rc))


def test_script(
    row: dict[str, str],
    endpoint: dict[str, object] | None = None,
) -> str:
    rc = row["RC"].strip()
    codes = expected_codes(rc)
    assertion = (endpoint or {}).get("assertion", {})
    if not isinstance(assertion, dict):
        assertion = {}
    success_regex = assertion.get("success_body_regex")
    failure_regex = assertion.get("failure_body_regex")
    extract = (endpoint or {}).get("extract", {})
    if not isinstance(extract, dict):
        extract = {}
    lines = [
        f"// {row['TC ID']} expected outcome: RC={rc}"
        + (" (NPCI ack pair: payer-payee)" if len(codes) > 1 else ""),
        "pm.test('HTTP status is 200', function () {",
        "  pm.response.to.have.status(200);",
        "});",
        "const bodyText = pm.response.text();",
        f"const expectedCodes = {json.dumps(codes)};",
    ]
    if success_expected(rc):
        pattern = success_regex or 'SUCCESS|"respCode"\\s*:\\s*"00"|"rc"\\s*:\\s*"00"'
        lines += [
            "pm.test('business result indicates success', function () {",
            f"  pm.expect(/{pattern}/i.test(bodyText)).to.be.true;",
            "});",
        ]
    else:
        if failure_regex:
            lines += [
                "pm.test('expected failure outcome is present in the response', function () {",
                f"  pm.expect(/{failure_regex}/i.test(bodyText), 'response should indicate failure').to.be.true;",
                "});",
            ]
        else:
            lines += [
                "pm.test('expected NPCI failure code is present in the response', function () {",
                "  const found = expectedCodes.some((code) => code !== '00' && bodyText.includes(code));",
                "  pm.expect(found, 'response should echo expected code ' + expectedCodes.join('|')).to.be.true;",
                "});",
            ]
    if extract:
        lines += ["// auto-extraction: make response ids available to later requests"]
        for env_key, path in extract.items():
            path_str = str(path)
            url_encode = False
            if path_str.endswith("@urlencode"):
                # capture with percent-encoding applied at CAPTURE time (the
                # encoded value is what downstream GET query interpolation
                # expects — e.g. smsContent containing spaces/&/*/() ).
                url_encode = True
                path_str = path_str[: -len("@urlencode")]
            if path_str == "@text":
                # capture the whole body; when the body IS a JSON string
                # (e.g. "NPCI,20150822,2.0|..."), capture its parsed value.
                lines += [
                    f"pm.test('captures {env_key} when present', function () {{",
                    "  try {",
                    "    let value = pm.response.text();",
                    "    try { const parsed = JSON.parse(value); if (typeof parsed === 'string') value = parsed; } catch (e) {}",
                    f"    if (value) pm.environment.set({json.dumps(env_key)}, value);",
                    "  } catch (e) {}",
                    "});",
                ]
                continue
            block = [
                f"pm.test('captures {env_key} when present', function () {{",
                "  try {",
                "    const parsed = pm.response.json();",
                f"    let value = {json.dumps(path_str)}.split('.').reduce((acc, part) => acc && acc[part], parsed);",
            ]
            if url_encode:
                block.append(
                    "    if (typeof value === 'string') value = encodeURIComponent(value);"
                )
            block += [
                f"    if (value !== undefined && value !== null && value !== '') pm.environment.set({json.dumps(env_key)}, String(value));",
                "  } catch (e) {}",
                "});",
            ]
            lines += block
    lines += [
        "pm.test('response envelope parses as JSON', function () {",
        "  pm.expect(() => pm.response.json()).to.not.throw;",
        "});",
    ]
    return "\n".join(lines)


def resolve_token(value: object, row: dict[str, str]) -> object:
    if not isinstance(value, str):
        if isinstance(value, dict):
            return {k: resolve_token(v, row) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_token(v, row) for v in value]
        return value
    if value == "__TEST_DATA_AS_VPA__":
        data = row["Test Data"].strip()
        return data if VPA_SHAPE.match(data) else "{{CUSTOMER_VPA}}"
    return value.replace("__TC_ID__", row["TC ID"])


def pick_family(mapping: dict[str, object], api_name: str) -> dict[str, object]:
    for family in mapping["families"]:
        if api_name in family["api_names"]:
            return family
    raise ValueError(
        f"no family routes API Name {api_name!r} — extend mapping.families"
    )


def pick_route(family: dict[str, object], row: dict[str, str]) -> dict[str, object]:
    description = row["Description"].lower()
    for route in family.get("routes", []):
        when = route.get("when", {})
        prefixes = when.get("test_case_id_prefix", [])
        if prefixes and not any(row["TC ID"].startswith(p) for p in prefixes):
            continue
        need = [d.lower() for d in when.get("description_all", [])]
        skip = [d.lower() for d in when.get("description_none", [])]
        any_terms = [d.lower() for d in when.get("description_any", [])]
        if (
            all(word in description for word in need)
            and not any(word in description for word in skip)
            and (not any_terms or any(word in description for word in any_terms))
        ):
            return route
    default_route = {"folder": family.get("folder"), "endpoint": family.get("endpoint")}
    if default_route["endpoint"] is None:
        raise ValueError(
            f"family {family['api_names']} row {row['TC ID']} matched no route and has no default"
        )
    return default_route


def build_request(
    row: dict[str, str],
    route: dict[str, object],
    endpoints: dict[str, dict[str, object]],
    overrides: dict[str, object],
) -> dict[str, object]:
    template = endpoints[route["endpoint"]]
    key_override = overrides.get("body") or {}
    rc = row["RC"].strip()
    name = (
        f"{row['TC ID']} [{','.join(expected_codes(rc))}] — {row['Description'][:72]}"
    )
    raw_path = str(template["path"])
    # endpoints may target a different service (e.g. the local CL cred service)
    base_url_var = str(template.get("base_url_var") or "BASE_URL")
    base_ref = "{{" + base_url_var + "}}"
    url: dict[str, object] = {
        "raw": base_ref + raw_path,
        "host": [base_ref],
    }
    if "?" in raw_path:
        path_part, query_part = raw_path.split("?", 1)
        url["path"] = path_part.lstrip("/").split("/")
        url["query"] = [
            {
                "key": pair.split("=", 1)[0],
                "value": pair.split("=", 1)[1] if "=" in pair else "",
            }
            for pair in query_part.split("&")
            if pair
        ]
    else:
        url["path"] = raw_path.lstrip("/").split("/")
    request: dict[str, object] = {
        "method": template["method"],
        "header": [
            {"key": key, "value": value, "type": "text"}
            for key, value in (template.get("headers") or {}).items()
        ],
        "url": url,
    }
    body = template.get("body")
    if isinstance(body, str):
        # raw verbatim body (XML/text payloads, e.g. the NPCI-simulator HBT row):
        # token resolution across the whole string; no key-level overrides.
        raw_body = body.replace("__TC_ID__", row["TC ID"]).replace(
            "__TEST_DATA_AS_VPA__", row["Test Data"].strip() or "{{CUSTOMER_VPA}}"
        )
        raw_language = (
            (template.get("raw_language") or "text")
            if isinstance(template, dict)
            else "text"
        )
        request["body"] = {
            "mode": "raw",
            "raw": raw_body,
            "options": {"raw": {"language": raw_language}},
        }
    elif body is not None:
        resolved = resolve_token(body, row)
        assert isinstance(resolved, dict)
        for key, value in (route.get("body") or {}).items():
            if value is None:
                resolved.pop(key, None)
            else:
                resolved[key] = resolve_token(value, row)
        for key, value in key_override.items():
            if value is None:
                resolved.pop(key, None)
            else:
                resolved[key] = resolve_token(value, row)
        request["body"] = {
            "mode": "raw",
            "raw": json.dumps(resolved, indent=2),
            "options": {"raw": {"language": "json"}},
        }
    return {
        "name": name,
        "request": request,
        "event": [
            {
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "exec": test_script(row, template).splitlines(),
                },
            }
        ],
        "_cz_manual": {
            "tc_id": row["TC ID"],
            "api_name": row["API Name"],
            "csv_test_data": row["Test Data"],
            "expected_rc": rc,
            "steps": row["Steps"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    mapping = json.loads(pathlib.Path(args.mapping).read_text())
    rows = list(csv.DictReader(pathlib.Path(args.csv).open()))
    endpoints = mapping["endpoints"]
    overrides_map = mapping.get("row_overrides", {})

    folders: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        family = pick_family(mapping, row["API Name"])
        route = pick_route(family, row)
        overrides = overrides_map.get(row["TC ID"], {})
        folder = route.get("folder") or family["api_names"][0]
        folders.setdefault(folder, []).append(
            build_request(row, route, endpoints, overrides)
        )

    collection = {
        "info": {
            "name": mapping["name"],
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            "description": mapping.get("description", ""),
        },
        "item": [
            {"name": folder, "item": items} for folder, items in sorted(folders.items())
        ],
        "event": [
            {
                "listen": "prerequest",
                "script": {
                    "type": "text/javascript",
                    "exec": (
                        COLLECTION_PREREQUEST
                        + "\n// computed variables (from mapping)\n"
                        + "\n".join(mapping.get("computed_vars", []))
                    ).splitlines(),
                },
            }
        ],
    }
    environment = {
        "name": mapping["name"] + " — environment (fill before use)",
        "values": [
            {"key": key, "value": default or "", "description": note, "enabled": True}
            for key, default, note in mapping["environment"]
        ],
        "_postman_variable_scope": "environment",
    }

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "-", mapping["name"].lower()).strip("-")
    (out_dir / f"{slug}.collection.json").write_text(
        json.dumps(collection, indent=2) + "\n"
    )
    (out_dir / f"{slug}.environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )

    notes = ["# " + mapping["name"] + " — build notes", ""]
    notes.append(f"- rows: {len(rows)}")
    notes.append(f"- folders: {len(folders)}")
    notes.append(f"- env keys: {len(environment['values'])}")
    notes.append("")
    notes.append("## Endpoint mapping")
    for family in mapping["families"]:
        for route in family.get("routes", [family]):
            endpoint = route.get("endpoint")
            if endpoint:
                spec = endpoints[endpoint].get("spec", "(no spec id)")
                notes.append(
                    f"- {family['api_names']} -> {endpoints[endpoint]['method']} "
                    f"{endpoints[endpoint]['path']} (folder: {route.get('folder')}, spec: {spec})"
                )
    dependencies = mapping.get("dependencies", [])
    if dependencies:
        notes.append("")
        notes.append("## Row dependencies (who must act for each case to pass)")
        notes.append("")
        notes.append("| TC | kind | reason |")
        notes.append("|---|---|---|")
        for dep in dependencies:
            notes.append(
                f"| {dep.get('tc', '?')} | {dep.get('kind', '?')} | {dep.get('reason', '')} |"
            )
    custom = mapping.get("notes_markdown", "")
    if custom:
        notes.append("")
        notes.append(custom)
    (out_dir / "NOTES.md").write_text("\n".join(notes) + "\n")
    print(f"rows={len(rows)} folders={len(folders)} -> {out_dir}")


if __name__ == "__main__":
    main()
