#!/usr/bin/env python3
"""Emit the LC-prefixed seed ladder SQL for a certification run.

Derives per-CSV-family VPA sets from the testcase sheet and any previously
proven device/account scaffolding, writes a single-BEGIN/COMMIT `seeds.sql`.

All rows start with LC- so the cleanup at handoff is one DELETE per table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

FAMILY_RANGE = {
    "BalEnq": range(13, 22),  # mt13..mt21
    "SetCre": range(22, 30),  # mt22..mt29
    "ChkTxn": range(30, 34),  # mt30..mt33
}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def emit_vpa(vpa: str, cust_id: str, mc_id: str, now: str) -> str:
    vid = f"LC-VPA-{vpa.split('@')[0].upper()[:16]:0<16}"
    return (
        f'INSERT INTO "Vpas" (id, vpa, "normalizedVpa", status, "CustomerId", "isPrimary", '
        f'"createdAt", "updatedAt", "MerchantCustomerId", active, "normalizedVpaHash", "vpaHash") '
        f"VALUES ('{vid}', '{vpa}', '{vpa}', 'ENABLED', '{cust_id}', false, '{now}', '{now}', "
        f"'{mc_id}', true, '{sha(vpa)}', '{sha(vpa)}') ON CONFLICT (vpa) DO NOTHING;"
    )


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--csv", required=True)
    a.add_argument("--merchant-db-id", required=True)
    a.add_argument("--mc-id", default="LC-0001")
    a.add_argument("--customer-id", default="LC-CUST-0001")
    a.add_argument("--account-id", default="LC-ACC-0001")
    a.add_argument("--account-hash", default="BAULOC0001")
    a.add_argument("--device-id", default="LC-DEV-0001")
    a.add_argument("--device-fingerprint", default="lcrawfp0001")
    a.add_argument("--device-ssid", default="1234")
    a.add_argument("--mobile", default="9876543210")
    a.add_argument("--vpa-domain", default="@vpa")
    a.add_argument("--out", required=True)
    opts = a.parse_args()

    now = (
        datetime.now(timezone(timedelta(hours=5, minutes=30)))
        .replace(microsecond=0)
        .strftime("%Y-%m-%d %H:%M:%S%z")
    )

    # read distinct VPAs the sheet references so we seed exactly the right set
    vps: set[str] = set()
    with open(opts.csv, newline="") as fh:
        for row in csv.DictReader(fh):
            td = (row.get("Test Data") or "").strip()
            if td and "@" in td and "," not in td:
                vps.add(td)
    # families whose VPAs the BCRP ladder proved deterministic
    for rng in FAMILY_RANGE.values():
        for i in rng:
            vps.add(f"mt{i}{opts.vpa_domain}")

    seeds: list[str] = [
        "-- certification-automation LC-scoped seeds (single transactional batch)",
        "BEGIN;",
    ]
    seeds.append(
        f'INSERT INTO "Customers" (id, "createdAt", "updatedAt", active, "mobileNumber") '
        f"VALUES ('{opts.customer_id}', '{now}', '{now}', true, '{opts.mobile}') ON CONFLICT (id) DO NOTHING;"
    )
    seeds.append(
        f'INSERT INTO "Devices" (id, os, fingerprint, ssid, model, version, manufacturer, '
        f'"createdAt", "updatedAt", "fingerprintHash", "ssidHash") VALUES ('
        f"'{opts.device_id}', 'android', '{opts.device_fingerprint}', '{opts.device_ssid}', "
        f"'UPICertDevice', '13', 'Juspay', '{now}', '{now}', "
        f"md5('{opts.device_fingerprint}'), md5('{opts.device_ssid}')) ON CONFLICT (id) DO NOTHING;"
    )
    seeds.append(
        f'INSERT INTO "MerchantCustomers" (id, "merchantCustomerId", "mobileNumber", "createdAt", '
        f'"updatedAt", active, "MerchantId", "CustomerId", secretKey, store, "packageName", email, '
        f'"mobileNumberHash", "DeviceId") VALUES ('
        f"'{opts.mc_id}', 'LOCAL-CERT-CUST-01', '{opts.mobile}', '{now}', '{now}', true, "
        f"'{opts.merchant_db_id}', '{opts.customer_id}', 'LOCALSECRET', '{{}}'::json, "
        f"'in.juspay.merchant', 'local@cert.dev', '{sha(opts.mobile)}', '{opts.device_id}') "
        f"ON CONFLICT (id) DO NOTHING;"
    )
    seeds.append(
        f'INSERT INTO "Accounts" (id, "accountNumber", "maskedAccountNumber", ifsc, "mpinSet", '
        f'"aadharEnabled", type, name, "bankCode", "bankName", "CustomerId", "accountHash", '
        f'"credsAllowed", "createdAt", "updatedAt", active, "default", "accSubType") '
        f"VALUES ('{opts.account_id}', '1234567890XX', 'xxxxxx', 'AABC0019', false, false, "
        f"'SAVINGS', 'LOCAL CERT CUSTOMER', 'AABC', 'NPCI MOCK BANK', '{opts.customer_id}', "
        f"'{opts.account_hash}', '[COMP]', '{now}', '{now}', true, true, 'SAVING') "
        f"ON CONFLICT (id) DO NOTHING;"
    )
    seeds.append(
        f'INSERT INTO "MerchantCustomerAccounts" (id, "AccountId", "MerchantCustomerId", "default", '
        f'"createdAt", "updatedAt", active, "accountHash") VALUES ('
        f"'LC-MCA-0001', '{opts.account_id}', '{opts.mc_id}', true, '{now}', '{now}', true, "
        f"'{opts.account_hash}') ON CONFLICT (id) DO NOTHING;"
    )
    for vpa in sorted(vps):
        seeds.append(emit_vpa(vpa, opts.customer_id, opts.mc_id, now))
    seeds.append(
        'INSERT INTO "Configurations" ("key","value","createdAt","updatedAt") VALUES '
        "('fallbackToMobileNumberForVpaResolution','true','%s','%s') "
        'ON CONFLICT ("key") DO UPDATE SET "value"=\'true\';' % (now, now)
    )
    seeds.append(
        f"UPDATE \"Merchants\" SET store=(COALESCE(store::jsonb,'{{}}'::jsonb) || "
        f'\'{{"vpaDomain": "{opts.vpa_domain}"}}\'::jsonb)::json '
        f"WHERE id='{opts.merchant_db_id}';"
    )
    seeds.append("COMMIT;")

    out_p = Path(opts.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(seeds) + "\n")
    print(f"seeds written -> {out_p} ({len(vps)} vpa rows, single transaction)")

    meta = {
        "generatedAt": now,
        "vpa_domain": opts.vpa_domain,
        "vpas": sorted(vps),
        "device_fingerprint_request_value": sha(
            opts.device_fingerprint + opts.device_ssid
        ),
        "merchant_db_id": opts.merchant_db_id,
    }
    meta_p = out_p.with_suffix(".meta.json")
    meta_p.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"meta          -> {meta_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
