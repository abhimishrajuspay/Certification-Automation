#!/usr/bin/env python3
"""Failure triage for certification-automation replay runs.

Given a marker (the request's upiRequestId, x-request-id, or the item name),
connects the dots across the three observability surfaces the BCRP ladder
teaches us to use:

  * /tmp/newton-local.log   (service/log shipping, `category` + `requestId` correlation)
  * redis (`newton-*`)      (request scope configuration / response caches)
  * psql (LC- prefixed rows) (immovable state seeds must have created)

Usage:

  python diagnose.py <marker> [--db CONN] [--redis-cmd redis-cli] \
      --log /tmp/newton-local.log [--mail] [--clear]

Without `--clear`, READ-ONLY. With `--clear`, invalidates the merchant /
merchant-customer / merchant-config redis keys after printing them (rarely needed
  - this skill's before-db-fix rule is taught by `reference/recipes.md` §6).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LOG_CATEGORIES = (
    "DB",
    "API",
    "Service",
    "BusinessLogic",
    "Generic",
    "Auth",
    "Warning",
)


def _sh(*cmd: str, timeout: int = 8) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def parse_log(marker: str, log_path: Path) -> dict:
    if not log_path.exists():
        return {"found": False, "log_path": str(log_path)}
    text = log_path.read_bytes().replace(b"\x00", b"").decode("utf-8", "ignore")
    lines = text.splitlines()
    hits: list[dict[str, str]] = []
    for line in lines:
        if marker.lower() not in line.lower():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        hits.append(
            {
                "level": str(rec.get("level")),
                "category": str(rec.get("category")),
                "label": str(rec.get("label"))[:120],
                "value": str(rec.get("value"))[:260],
            }
        )
    # newton is chatty; report only the weird ones last-to-first
    errors = [h for h in hits if h["level"] in ("Error", "Exception", "Warning")]
    return {
        "found": bool(hits),
        "hits": len(hits),
        "errors": errors[-6:],
        "errors_total": len(errors),
    }


def list_redis(cli: str) -> list[str]:
    rc, out, err = _sh(cli, "--scan", "--pattern", "newton-*")
    if rc != 0:
        return []
    return sorted(set(filter(None, out.splitlines())))


def psql_probe(conn: str) -> dict[str, list[str]]:
    probes = {
        "customer_chain": (
            'SELECT mc."merchantCustomerId", mc.active, mc."DeviceId", '
            'c."mobileNumber" FROM "MerchantCustomers" mc '
            'JOIN "Customers" c ON c.id = mc."CustomerId" '
            "WHERE mc.\"merchantCustomerId\" = 'LOCAL-CERT-CUST-01';"
        ),
        "accounts": (
            'SELECT id, "accountHash", "maskedAccountNumber", active '
            "FROM \"Accounts\" WHERE id LIKE 'LC-%' LIMIT 5;"
        ),
        "linked": (
            'SELECT mca."MerchantCustomerId", mca."AccountId", mca.active '
            "FROM \"MerchantCustomerAccounts\" mca WHERE id LIKE 'LC-%' LIMIT 5;"
        ),
        "merchant_cfg": (
            'SELECT "MerchantId", active, "checksumBypassId", "upiVpa", '
            '"fundOutConstraint" FROM "MerchantConfigurations" LIMIT 5;'
        ),
        "config_row": (
            'SELECT "key","value" FROM "Configurations" '
            "WHERE \"key\" IN ('fallbackToMobileNumberForVpaResolution');"
        ),
    }
    out: dict[str, list[str]] = {}
    for name, q in probes.items():
        rc, o, _e = _sh(
            "psql",
            conn,
            "-t",
            "-A",
            "-F;",
            "-c",
            q,
        )
        out[name] = [r.strip() for r in o.splitlines() if r.strip()] if rc == 0 else []
    return out


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("marker")
    a.add_argument("--db", required=True)
    a.add_argument("--redis-cmd", default="redis-cli")
    a.add_argument("--log", default="/tmp/newton-local.log")
    a.add_argument("--clear", action="store_true")
    opts = a.parse_args()

    print(f"=== diagnose(marker={opts.marker})")

    log_result = parse_log(opts.marker, Path(opts.log))
    print("\n[1/3] newton-local.log")
    if not log_result.get("found"):
        print(f"  no entries matching marker ({log_result})")
    else:
        print(
            f"  hits={log_result['hits']} errors={log_result['errors_total']} "
            f"(last {len(log_result['errors'])} shown)"
        )
        for e in log_result["errors"]:
            print(f"  [{e['level']:9s}] {e['category']:14s}] {e['label']}")
            print(f"      {e['value']}")

    print("\n[2/3] redis keys (read-only unless --clear):")
    keys = list_redis(opts.redis_cmd)
    interesting = [
        k
        for k in keys
        if any(t in k for t in ("merchant", "config", "customer", "txn"))
    ]
    for k in interesting[:18]:
        print(f"  {k}")
    if len(interesting) > 18:
        print(f"  … {len(interesting) - 18} more")
    if opts.clear:
        to_clear = [
            k
            for k in keys
            if (
                "TESTMERCHANT" in k
                or "LOCAL-CERT-CUST-01" in k
                or "configuration" in k.lower()
            )
        ]
        for k in to_clear:
            subprocess.run([opts.redis_cmd, "DEL", k], check=False)
            print(f"  deleted {k}")

    print("\n[3/3] psql (LC-scoped probes)")
    probes = psql_probe(opts.db)
    for name, rows in probes.items():
        print(f"  {name:14s}: {rows or '[no LC- rows]'}")
    return 0 if not opts.clear else 0


if __name__ == "__main__":
    sys.exit(main())
