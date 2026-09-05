#!/usr/bin/env python3
"""Fetch full endpoint specifications from a local Newton MCP server.

Run from the Certification-Automation repo root with the project venv:

  venv/bin/python3 .opencode/skills/certification-postman/scripts/fetch_specs.py \
    --mcp-url http://localhost:8000/mcp \
    --endpoint-id newton.s2s.post.merchants.vpas.validity \
    --out-dir specs/

Writes one markdown file per endpoint id into --out-dir so the audit and the
collection build can cite the exact spec used. Prints each fetched id, its
character count, and the first "calls NPCI ..."-style line when present.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

# .opencode/skills/certification-postman/scripts/x.py -> repo root is 4 levels up
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from grounding.mcp import MCPClient, MCPClientConfig  # noqa: E402


async def _fetch(mcp_url: str, endpoint_ids: list[str], out_dir: pathlib.Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = MCPClient(
        MCPClientConfig(endpoint=mcp_url, allowed_tools=("get_api_spec",))
    )
    await client.connect()
    failures = 0
    for endpoint_id in endpoint_ids:
        result = await client.call_tool("get_api_spec", {"endpoint_id": endpoint_id})
        text = "".join(result.text_blocks)
        if text.startswith("❌") or "not found" in text[:200].lower():
            failures += 1
            print(f"MISSING {endpoint_id}")
            continue
        (out_dir / f"{endpoint_id}.md").write_text(text)
        npci = next(
            (line.strip()[:120] for line in text.splitlines() if "NPCI" in line),
            "-",
        )
        print(f"OK {endpoint_id} chars={len(text)} npci_ref={npci}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-url", default="http://localhost:8000/mcp")
    parser.add_argument("--endpoint-id", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    failures = asyncio.run(
        _fetch(args.mcp_url, args.endpoint_id, pathlib.Path(args.out_dir))
    )
    if failures:
        sys.exit(f"{failures} endpoint id(s) missing; fix candidates and rerun")


if __name__ == "__main__":
    main()
