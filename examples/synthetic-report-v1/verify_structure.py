"""Check row/value associations in Docling JSON for this authored fixture only."""

import argparse
import json
from pathlib import Path

EXPECTED = (
    ("North", "120,000", "135,000"),
    ("South", "98,500", "102,300"),
    ("East", "87,200", "91,000"),
)


def check_structure(document: dict) -> list[str]:
    rows = []
    for table in document.get("tables", []):
        cells = table.get("data", {}).get("table_cells", [])
        by_row = {}
        for cell in cells:
            row = cell["start_row_offset_idx"]
            column = cell["start_col_offset_idx"]
            by_row.setdefault(row, {})[column] = cell.get("text", "").strip()
        rows.extend(tuple(values[c] for c in sorted(values)) for values in by_row.values())
    return [f"Missing associated row: {' | '.join(row)}" for row in EXPECTED if row not in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docling_json", type=Path)
    args = parser.parse_args()
    failures = check_structure(json.loads(args.docling_json.read_text(encoding="utf-8")))
    if failures:
        raise SystemExit("\n".join(failures))
    print("All three authored row/value associations are present; not an OCR accuracy benchmark.")


if __name__ == "__main__":
    main()
