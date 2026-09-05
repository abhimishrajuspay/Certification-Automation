#!/usr/bin/env python3
"""Replay a Postman collection against the LOCAL newton-hs dev server.

BYPASSES signature/auth via development-only headers:
  x-bypass-merchant-checksum: <checksumBypassId>  (MerchantConfigurations row)
  x-bypass-response-encryption: true              (plain body back)

Env substitution order:
  --env-file kv.json | inline --set K=V
  BASE_URL/API_VERSION are overridden unless env file supplies them.

REQUEST_ID behavior:
  - not passed via env/inline: derived per item as LC{TC_ID}_{epoch_ms}
    (guards against narrowly-scoped idempotency)
  - passed explicitly: propagated verbatim (allows ChkTxn-chaining flows)

Writes per-item result rows to the given output JSONL path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

VAR_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def flatten(collection: dict):
    for folder in collection.get("item", []):
        folder_name = folder.get("name", "")
        for leaf in folder.get("item", []):
            if "item" in leaf:  # nested folder: descend one level
                for sub in leaf.get("item", []):
                    yield folder_name, leaf.get("name", ""), sub
            else:
                yield folder_name, "", leaf


def substitute(text: str, env: dict, missing: set) -> str:
    def repl(m):
        key = m.group(1)
        if key in env:
            return str(env[key])
        missing.add(key)
        return m.group(0)

    return VAR_RE.sub(repl, text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--environment", default=None)
    ap.add_argument("--set", action="append", default=[], help="K=V inline env")
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--only", default=None, help="substring filter on item name")
    ap.add_argument("--folder", default=None, help="substring filter on folder name")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument(
        "--base-url",
        default=None,
        help=(
            "explicit gateway override; when omitted the environment's"
            " BASE_URL wins (fails loudly when neither provides one)"
        ),
    )
    ap.add_argument("--api-version", default="x2")
    ap.add_argument("--merchant-id", default="TESTMERCHANT")
    ap.add_argument("--channel-id", default="TESTMERCHANT")
    ap.add_argument("--bypass-id", default="localcert")
    args = ap.parse_args()

    env: dict[str, str] = {}
    if args.environment:
        doc = json.load(open(args.environment))
        for e in doc.get("values", []):
            if e.get("enabled", True):
                v = e.get("value") or ""
                if v != "":
                    env[e["key"]] = v
    for kv in args.set:
        k, _, v = kv.partition("=")
        env[k] = v
    if args.base_url:
        # Explicit operator override always wins over the environment file.
        env["BASE_URL"] = args.base_url
    if not env.get("BASE_URL"):
        sys.exit(
            "replay_local: no BASE_URL resolved — pass --base-url or set a"
            " non-empty BASE_URL value in the environment file"
        )
    env["API_VERSION"] = args.api_version
    env.setdefault("MERCHANT_ID", args.merchant_id)
    env.setdefault("MERCHANT_CHANNEL_ID", args.channel_id)

    coll = json.load(open(args.collection))
    rows = []
    missing_all: set[str] = set()
    for folder, group, item in flatten(coll):
        if args.folder and args.folder not in folder:
            continue
        if args.only and args.only not in item.get("name", ""):
            continue
        req = item.get("request") or {}
        if isinstance(req, str):
            continue  # no request recorded
        url = ((req.get("url") or {}).get("raw")) or ""
        method = req.get("method", "GET")
        body = req.get("body") or {}
        raw = body.get("raw") if isinstance(body, dict) else None
        tc_id = (item.get("name", "") or "").split(" ")[0]
        local_env = dict(env)
        # Per-item request identity so replays never collide on redis idempotency /
        # Transactions uniques. If the operator passes --set REQUEST_ID=..., that
        # value wins (chain-style replays); otherwise derive LC{tc}_{epoch_ms}.
        if "REQUEST_ID" not in local_env and tc_id:
            local_env["REQUEST_ID"] = f"LC{tc_id}_{int(time.time() * 1000)}"
        missing: set[str] = set()
        url_s = substitute(url, local_env, missing)
        body_s = substitute(raw, local_env, missing) if raw else None
        missing_all |= missing
        headers = {
            "content-type": "application/json",
            "x-merchant-id": env.get("MERCHANT_ID", args.merchant_id),
            "x-merchant-channel-id": env.get("MERCHANT_CHANNEL_ID", args.channel_id),
            "x-bypass-merchant-checksum": args.bypass_id,
            "x-bypass-response-encryption": "true",
            "x-timestamp": "2026-08-31T16:30:00.000Z",
            "x-merchant-checksum": "bypass",
        }
        # let collection-authored headers layering in (they may pin content-type/others)
        for h in req.get("header") or []:
            if h.get("disabled"):
                continue
            headers[h["key"]] = substitute(h.get("value", ""), env, missing)

        data = body_s.encode() if (method != "GET" and body_s is not None) else None
        r = urllib.request.Request(url_s, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r, timeout=args.timeout) as resp:
                status = resp.status
                resp_body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:  # non-200
            status = e.code
            resp_body = e.read().decode("utf-8", "replace")[:2000]
        except BaseException as e:
            status = -1
            resp_body = f"TRANSPORT_ERROR {type(e).__name__}: {e}"[:400]

        row = {
            "folder": folder,
            "group": group,
            "name": item.get("name", "")[:90],
            "method": method,
            "path": url_s.replace(env.get("BASE_URL", ""), "")[:160],
            "missing_vars": sorted(missing),
            "http": status,
            "body": (body_s[:400] if body_s else None),
            "resp": resp_body[:600],
        }
        rows.append(row)
        print(f"[{status}] {item.get('name', '')[:70]} :: {row['path']}", flush=True)

    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n== {len(rows)} requests -> {args.out}")
    if missing_all:
        print("== unresolved vars:", ", ".join(sorted(missing_all)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
