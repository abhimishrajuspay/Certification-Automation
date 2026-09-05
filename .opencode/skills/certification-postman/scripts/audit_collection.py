#!/usr/bin/env python3
"""Audit a generated certification collection. Exit nonzero on ANY failure.

Checks (in order):
  1. JSON validity, Collection v2.1 schema URL, folder/item structure.
  2. Every CSV row renders exactly once (by TC ID) — no drops, no duplicates.
  3. Placeholder coverage: every {{VAR}} in the collection resolves to an
     environment key or a pre-request-computed variable (IAT, REQUEST_ID).
  4. Mandatory-field conformance: for each endpoint template in the mapping,
     every `*`-required field in its spec markdown (## Request Schema) appears
     in the template body (dot-key paths flattened, e.g. data.ki).
  5. node --check passes for the collection pre-request + every test script
     (skipped with a warning when `node` is unavailable).

Usage:
  venv/bin/python3 .opencode/skills/certification-postman/scripts/audit_collection.py \
    --collection out/x.collection.json --environment out/x.environment.json \
    --csv cases.csv --mapping mapping.json --spec-dir specs/
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

COMPUTED_VARS = {"IAT", "REQUEST_ID", "__RESOLVED_BODY"}


def fail(message: str) -> None:
    print(f"AUDIT-FAIL {message}")
    sys.exit(1)


def flatten_keys(body: object, prefix: str = "") -> set[str]:
    if isinstance(body, dict):
        keys: set[str] = set()
        for key, value in body.items():
            keys.add(prefix + key)
            keys |= flatten_keys(value, prefix + key + ".")
        return keys
    return set()


def mandatory_fields(spec_text: str) -> set[str]:
    if "## Request Schema" not in spec_text:
        return set()
    schema = spec_text.split("## Request Schema", 1)[1]
    schema = schema.split("## Response Schema", 1)[0].split("## Examples", 1)[0]
    required = set()
    for line in schema.splitlines():
        match = re.match(r"- \*\*([A-Za-z0-9_.]+)\*\*\*", line)
        if match:
            required.add(match.group(1))
    return required


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--spec-dir", required=True)
    args = parser.parse_args()

    collection = json.loads(pathlib.Path(args.collection).read_text())
    environment = json.loads(pathlib.Path(args.environment).read_text())
    mapping = json.loads(pathlib.Path(args.mapping).read_text())
    rows = list(csv.DictReader(pathlib.Path(args.csv).open()))

    # 1. structure
    if not collection["info"]["schema"].endswith("v2.1.0/collection.json"):
        fail("info.schema is not the official Collection v2.1 URL")
    items = [item for folder in collection["item"] for item in folder["item"]]
    if not items:
        fail("collection has no items")

    # 2. row coverage
    csv_ids = [row["TC ID"] for row in rows]
    rendered_ids = [item["_cz_manual"]["tc_id"] for item in items]
    missing = sorted(set(csv_ids) - set(rendered_ids))
    extra = sorted(set(rendered_ids) - set(csv_ids))
    if missing or extra:
        fail(f"row coverage mismatch missing={missing} extra={extra}")
    if len(set(rendered_ids)) != len(rendered_ids):
        fail("duplicate TC IDs rendered")

    # 3. placeholder coverage
    env_keys = {entry["key"] for entry in environment["values"]} | COMPUTED_VARS
    used = set(re.findall(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}", json.dumps(collection)))
    stray = sorted(used - env_keys)
    if stray:
        fail(f"placeholders without environment/computed source: {stray}")

    # 4. mandatory fields vs spec evidence
    spec_dir = pathlib.Path(args.spec_dir)
    problems = []
    for key, template in mapping["endpoints"].items():
        spec = template.get("spec")
        if not spec or template.get("body") is None:
            continue
        spec_file = spec_dir / f"{spec}.md"
        if not spec_file.exists():
            problems.append(f"{key}: spec evidence missing {spec_file}")
            continue
        required = mandatory_fields(spec_file.read_text())
        have = flatten_keys(template["body"])
        missing = sorted(required - have)
        if missing:
            problems.append(f"{key}: mandatory fields missing from template: {missing}")
    if problems:
        for problem in problems:
            print("AUDIT-FAIL", problem)
        sys.exit(1)

    # 5. script syntax
    scripts = {
        "\n".join(event["script"]["exec"]) for item in items for event in item["event"]
    }
    scripts.add("\n".join(collection["event"][0]["script"]["exec"]))
    node = shutil.which("node")
    if node is None:
        print("AUDIT-WARN node not available; skipping script syntax check")
    else:
        for script in scripts:
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
                handle.write(script)
            result = subprocess.run([node, "--check", handle.name], capture_output=True)
            if result.returncode != 0:
                fail(f"script syntax error: {result.stderr.decode()[:300]}")

    print(
        f"AUDIT-PASS items={len(items)} folders={len(collection['item'])} "
        f"env_keys={len(environment['values'])} scripts={len(scripts)}"
    )


if __name__ == "__main__":
    main()
