import json
from pathlib import Path

import pytest

import ocr_llm_pipeline as pipeline


def prepare(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source = input_dir / "sample.pdf"
    source.write_bytes(b"synthetic input")
    monkeypatch.setattr(pipeline, "build_converter", lambda: object())
    calls = []

    def convert(file_path, converter, output_dir, assets, output_stem):
        calls.append(Path(file_path).read_bytes())
        (Path(output_dir) / f"{output_stem}.md").write_text("converted")
        return "converted", {"text": "converted"}

    monkeypatch.setattr(pipeline, "run_ocr_pipeline", convert)
    return input_dir, source, tmp_path / "output", calls


def test_resume_skips_only_unchanged_complete_outputs(tmp_path, monkeypatch):
    inputs, source, output, calls = prepare(tmp_path, monkeypatch)
    pipeline.process_documents(inputs, output, resume=True)
    pipeline.process_documents(inputs, output, resume=True)
    assert len(calls) == 1
    source.write_bytes(b"changed input")
    pipeline.process_documents(inputs, output, resume=True)
    assert len(calls) == 2
    (output / "sample.json").write_text("tampered")
    pipeline.process_documents(inputs, output, resume=True)
    assert len(calls) == 3


def test_llm_failure_does_not_mark_partial_artifacts_complete(tmp_path, monkeypatch):
    inputs, source, output, calls = prepare(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic provider failure")

    monkeypatch.setattr(pipeline, "run_llm_extraction", fail)
    with pytest.raises(RuntimeError):
        pipeline.process_documents(inputs, output, tmp_path / "reports", run_llm=True, resume=True)
    manifest = json.loads((output / "pipeline-status.json").read_text())
    assert manifest["files"]["sample.pdf"]["state"] == "failed"
    monkeypatch.setattr(pipeline, "run_llm_extraction", lambda *args, **kwargs: "report")
    pipeline.process_documents(inputs, output, tmp_path / "reports", run_llm=True, resume=True)
    assert len(calls) == 2
    assert json.loads((output / "pipeline-status.json").read_text())["files"]["sample.pdf"]["state"] == "complete"


@pytest.mark.parametrize("names", [
    ["pipeline-status.pdf"],
    ["PIPELINE-STATUS.png", "pipeline-status.pdf", "pipeline-status_pdf.pdf"],
])
def test_status_sidecar_never_overwrites_document_json(tmp_path, monkeypatch, names):
    inputs, source, output, calls = prepare(tmp_path, monkeypatch)
    source.unlink()
    for name in names:
        (inputs / name).write_bytes(b"synthetic input")
    pipeline.process_documents(inputs, output, resume=True)
    manifest = json.loads((output / "pipeline-status.json").read_text())
    outputs = list(output.glob("*.json"))
    assert len(outputs) == len(names) + 1
    assert sum(json.loads(p.read_text()) == {"text": "converted"} for p in outputs) == len(names)
    assert all(record["state"] == "complete" for record in manifest["files"].values())
    pipeline.process_documents(inputs, output, resume=True)
    assert len(calls) == len(names)
