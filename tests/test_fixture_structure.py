import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1] / "examples" / "synthetic-report-v1"
spec = importlib.util.spec_from_file_location("fixture_structure", ROOT / "verify_structure.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_captured_flattened_table_is_rejected():
    document = json.loads((ROOT / "captured-output" / "docling-2.130.0.json").read_text())
    assert len(verifier.check_structure(document)) == 3


def test_all_numbers_without_row_associations_are_rejected():
    document = {"texts": [{"text": "North South East 120,000 135,000 98,500 102,300 87,200 91,000"}]}
    assert verifier.check_structure(document)


def test_authored_structured_reference_and_permuted_values():
    cells = [{"start_row_offset_idx": row, "start_col_offset_idx": col, "text": text}
             for row, values in enumerate(verifier.EXPECTED) for col, text in enumerate(values)]
    document = {"tables": [{"data": {"table_cells": cells}}]}
    assert verifier.check_structure(document) == []
    cells[1]["text"], cells[2]["text"] = cells[2]["text"], cells[1]["text"]
    assert len(verifier.check_structure(document)) == 1
