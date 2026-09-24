#!/usr/bin/env python3
"""Intake for the certification-automation skill.

Asks the exact questions the skill needs ONCE, validates reachability of the
repository, mocker, DB and redis, and writes out/<run-id>/intake.json.

Intended usage (human-in-the-loop at first):

  python scripts/intake.py --run-id bcrp-01 --csv "Test-case-BCRP-Comfort-v1 - Sheet1 copy.csv" \
      --collection Newton.postman_collection.json \
      --repo /Users/abhishek.mishra/Desktop/repos/newton-hs \
      --mocker /Users/abhishek.mishra/Desktop/repos/npci-mocking \
      --db 'postgresql://abhishek.mishra@localhost:5432/amazon-upi' \
      --redis redis-cli --merchant TESTMERCHANT

Anything misspecified ends up as intake errors, which the next phase
(plan) treats as gates, so a teleprinted failure is a fix BEFORE any DB
or network mutation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]


def _sh(*cmd: str, timeout: int = 8) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _find_dirs(root: Path, needle: str) -> list[str]:
    return [p.name for p in root.iterdir() if p.is_dir() and needle in p.name]


def _check_collection(p: Path) -> dict:
    if not p.exists():
        return {"ok": False, "reason": f"missing {p}"}
    try:
        data = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"not JSON: {exc}"}
    items = data.get("item", [])
    return {"ok": True, "items": len(items)}


def _check_repo(p: Path) -> dict:
    checks = {}
    if not (p / ".git").exists():
        checks["git"] = "missing .git"
    for cand in ("src/Newton", "db_schema", ".env.local"):
        if (p / cand).exists():
            checks[cand] = "ok"
        else:
            checks[cand] = "missing"
    return checks


def _check_mocker(p: Path, port: int) -> dict:
    out: dict[str, object] = {"package_json": (p / "package.json").exists()}
    conf = p / "src" / "config.ts"
    if conf.exists():
        text = conf.read_text()
        out["defaults_present"] = {
            k: k in text
            for k in ("PSP_IP_META_APL", "PSP_PORT", "pspDetails", "cbsDetails")
        }
    try:
        http = _sh(
            "curl",
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            f"http://localhost:{port}/",
        )
        out["reachable"] = http[2]  # any code number wins
    except Exception:  # noqa: BLE001
        out["reachable"] = "unreachable"
    return out


def _check_db(conn: str) -> dict:
    q = 'SELECT 1 FROM "Merchants" WHERE id IS NOT NULL LIMIT 1;'
    try:
        rc, out, err = _sh("psql", conn, "-tAc", q, timeout=10)
    except FileNotFoundError:
        return {"ok": False, "reason": "psql missing"}
    return {"ok": rc == 0 and out.strip() == "1", "error": err.strip() or None}


def _check_redis(cli: str) -> dict:
    try:
        rc, out, err = _sh(cli, "PING", timeout=6)
    except FileNotFoundError:
        return {"ok": False, "reason": "redis-cli missing"}
    return {"ok": rc == 0 and out.strip() == "PONG", "error": err.strip() or None}


def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--run-id", required=True)
    a.add_argument("--csv", required=True)
    a.add_argument("--collection", required=True)
    a.add_argument("--repo", required=True)
    a.add_argument("--mocker", required=True)
    a.add_argument("--db", required=True)
    a.add_argument("--redis", default="redis-cli")
    a.add_argument("--merchant", required=True)
    a.add_argument(
        "--dummy-collection",
        default="",
        help="Reference Postman collection carrying cred-generation logic (createCredBlock helper)",
    )
    a.add_argument(
        "--run-after-build",
        choices=("yes", "no"),
        default="yes",
        help="Immediately run replay_local.py ladder after the collection is built (default: yes)",
    )
    a.add_argument("--out", default="out")
    opts = a.parse_args()

    out_dir = SKILL / opts.out / opts.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs: dict[str, object] = {}
    validation: dict[str, object] = {}
    record: dict[str, object] = {
        "run_id": opts.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "validation": validation,
        "status": "pending",
    }
    csv_p = Path(opts.csv).expanduser().resolve()
    col_p = Path(opts.collection).expanduser().resolve()
    repo_p = Path(opts.repo).expanduser().resolve()
    mocker_p = Path(opts.mocker).expanduser().resolve()
    dummy_p = (
        Path(opts.dummy_collection).expanduser().resolve()
        if opts.dummy_collection
        else None
    )

    inputs["csv"] = str(csv_p)
    inputs["collection"] = str(col_p)
    inputs["repo"] = str(repo_p)
    inputs["mocker"] = str(mocker_p)
    inputs["merchant"] = opts.merchant
    inputs["run_after_build"] = opts.run_after_build == "yes"
    if dummy_p is not None:
        inputs["dummy_collection"] = str(dummy_p)

    validation["csv"] = {
        "ok": csv_p.exists(),
        "rows": max(0, sum(1 for _ in csv_p.open()) - 1) if csv_p.exists() else 0,
    }
    validation["collection"] = _check_collection(col_p)
    if dummy_p is not None:
        validation["dummy_collection"] = _check_collection(dummy_p)
    validation["repo"] = _check_repo(repo_p)
    validation["mocker"] = _check_mocker(mocker_p, 8089)
    validation["db"] = _check_db(opts.db)
    validation["redis"] = _check_redis(opts.redis)

    ok = all(
        v.get("ok")
        for v in (
            validation["csv"],
            validation["collection"],
            validation["db"],
            validation["redis"],
        )
        if isinstance(v, dict)
    )
    if ok:
        mocker_ch = validation["mocker"]
        if isinstance(mocker_ch, dict):
            repo_ok = all(v == "ok" for k, v in validation["repo"].items())
        else:
            repo_ok = False
        ok = repo_ok and mocker_ch.get("package_json") == True  # noqa: E712

    if ok:
        record["status"] = "ready"
    else:
        record["status"] = "failed-blockers"

    (out_dir / "intake.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"intake: {record['status']} -> {out_dir}")
    for slot, ver in validation.items():
        print(f"  {slot:12s}: {json.dumps(ver)[:120]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
