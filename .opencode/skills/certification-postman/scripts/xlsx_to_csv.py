#!/usr/bin/env python3
"""Minimal stdlib-only XLSX -> CSV converter for certification sheets.

Usage:
  venv/bin/python3 <skill>/scripts/xlsx_to_csv.py INPUT.xlsx [-o output.csv] [--sheet NAME]

Handles the constructs real certification sheets use: shared strings, inline
strings, numeric/general cells, sparse rows, and multi-sheet workbooks
(default: the first sheet, or --sheet by name). No merged-cell magic — a row's
first non-empty row becomes the CSV header.
"""

from __future__ import annotations

import argparse
import csv
from html import unescape
import pathlib
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _column_index(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref).group(1)
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _shared_strings(handle: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(handle.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.iter(f"{NS}si"):
        parts = [t.text or "" for t in item.iter(f"{NS}t")]
        values.append(unescape("".join(parts)))
    return values


RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _sheet_path(handle: zipfile.ZipFile, sheet_name: str | None) -> str:
    workbook = ET.fromstring(handle.read("xl/workbook.xml"))
    sheets = workbook.find(f"{NS}sheets")
    assert sheets is not None
    entries = [(sheet.get("name"), sheet.get(f"{RID}id")) for sheet in sheets]
    relationships = ET.fromstring(handle.read("xl/_rels/workbook.xml.rels"))
    targets: dict[str, str] = {}
    for relation in relationships.iter(f"{REL}Relationship"):
        targets[relation.get("Id")] = relation.get("Target")

    def resolve(rid: str) -> str:
        target = targets.get(rid)
        if target is None:
            sys.exit(f"sheet relationship {rid!r} has no target")
        return "xl/" + target.lstrip("/") if not target.startswith("xl/") else target

    if sheet_name is not None:
        for name, rid in entries:
            if name == sheet_name:
                return resolve(rid)
        sys.exit(
            f"sheet {sheet_name!r} not found; available: {[n for n, _ in entries]}"
        )
    return resolve(entries[0][1])


def convert(xlsx_path: pathlib.Path, sheet_name: str | None) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path) as handle:
        shared = _shared_strings(handle)
        root = ET.fromstring(handle.read(_sheet_path(handle, sheet_name)))
    rows: list[list[str]] = []
    for row in root.iter(f"{NS}row"):
        cells: dict[int, str] = {}
        width = 0
        for cell in row.iter(f"{NS}c"):
            ref = cell.get("r", "")
            if not ref:
                continue
            column = _column_index(ref)
            width = max(width, column + 1)
            cell_type = cell.get("t")
            value_node = cell.find(f"{NS}v")
            inline_node = cell.find(f"{NS}is")
            if cell_type == "s" and value_node is not None:
                cells[column] = shared[int(value_node.text or 0)]
            elif cell_type == "inlineStr" and inline_node is not None:
                cells[column] = "".join(
                    t.text or "" for t in inline_node.iter(f"{NS}t")
                )
            elif value_node is not None:
                cells[column] = value_node.text or ""
        if cells:
            rows.append([cells.get(index, "") for index in range(width)])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path)
    parser.add_argument("--sheet")
    args = parser.parse_args()
    rows = convert(args.xlsx, args.sheet)
    if not rows:
        sys.exit("no rows extracted")
    output = args.output or args.xlsx.with_suffix(".csv")
    with output.open("w", newline="") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)
    print(f"rows={len(rows)} -> {output}")


if __name__ == "__main__":
    main()
