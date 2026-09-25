import json
from pathlib import Path

import numpy as np
import pytest
import cv2

from ocr_llm_pipeline import (
    get_ollama_api_key,
    iter_input_files,
    preprocess_image,
    process_documents,
    run_ocr_pipeline,
)


def test_iter_input_files_filters_by_extension_and_sorts(tmp_path):
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.pdf").write_bytes(b"")
    (tmp_path / "c.jpg").write_bytes(b"")
    (tmp_path / "ignored.txt").write_bytes(b"")
    (tmp_path / "subdir").mkdir()

    files = list(iter_input_files(tmp_path))

    assert [f.name for f in files] == ["a.pdf", "b.png", "c.jpg"]


def test_iter_input_files_on_empty_directory(tmp_path):
    assert list(iter_input_files(tmp_path)) == []


def test_preprocess_image_returns_a_binarized_temp_file(tmp_path):
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), image)

    output_path = preprocess_image(image_path)

    result = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    assert result is not None
    # adaptiveThreshold output is binary: only 0 and 255 values
    assert set(np.unique(result)).issubset({0, 255})
    Path(output_path).unlink()


def test_preprocess_image_uses_unique_temporary_paths(tmp_path):
    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), np.full((50, 50, 3), 200, dtype=np.uint8))

    first = preprocess_image(image_path)
    second = preprocess_image(image_path)
    assert first != second
    Path(first).unlink()
    Path(second).unlink()


def test_ocr_conversion_failure_removes_preprocessed_image(tmp_path):
    image_path = tmp_path / "input.png"
    cv2.imwrite(str(image_path), np.full((50, 50, 3), 200, dtype=np.uint8))

    class FailingConverter:
        converted_path = None

        def convert(self, path):
            self.converted_path = Path(path)
            assert self.converted_path.exists()
            raise RuntimeError("conversion failed")

    converter = FailingConverter()
    with pytest.raises(RuntimeError, match="conversion failed"):
        run_ocr_pipeline(image_path, converter, tmp_path / "output")
    assert converter.converted_path is not None
    assert not converter.converted_path.exists()


def test_preprocess_image_raises_on_unreadable_file(tmp_path):
    bad_path = tmp_path / "not_an_image.png"
    bad_path.write_bytes(b"not a real image")

    with pytest.raises(ValueError, match="Could not read image"):
        preprocess_image(bad_path)


def test_get_ollama_api_key_reads_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key-123")

    assert get_ollama_api_key() == "test-key-123"


def test_get_ollama_api_key_raises_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Ollama API key is not configured"):
        get_ollama_api_key()


def test_missing_report_directory_fails_before_converter_setup(tmp_path, monkeypatch):
    def unexpected_converter():
        raise AssertionError("Converter must not be initialized for invalid arguments")

    monkeypatch.setattr("ocr_llm_pipeline.build_converter", unexpected_converter)
    with pytest.raises(ValueError, match="reports_dir must be provided"):
        process_documents(tmp_path, tmp_path / "output", run_llm=True)
    assert not (tmp_path / "output").exists()


def test_process_documents_preserves_outputs_with_duplicate_input_stems(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("report.pdf", "report.png", "report_pdf.jpg"):
        (input_dir / name).write_bytes(b"fixture")

    monkeypatch.setattr("ocr_llm_pipeline.build_converter", lambda: object())

    def fake_ocr(file_path, converter, output_dir, assets, output_stem):
        markdown = Path(file_path).name
        (Path(output_dir) / f"{output_stem}.md").write_text(markdown, encoding="utf-8")
        return markdown, {"source": markdown}

    monkeypatch.setattr("ocr_llm_pipeline.run_ocr_pipeline", fake_ocr)
    monkeypatch.setattr("ocr_llm_pipeline.run_llm_extraction", lambda text, model_name: text)

    markdown_dir = tmp_path / "markdown"
    report_dir = tmp_path / "reports"
    process_documents(input_dir, markdown_dir, report_dir, run_llm=True)

    expected = {
        "report_pdf_2": "report.pdf",
        "report_png": "report.png",
        "report_pdf": "report_pdf.jpg",
    }
    for stem, source in expected.items():
        assert (markdown_dir / f"{stem}.md").read_text(encoding="utf-8") == source
        assert json.loads((markdown_dir / f"{stem}.json").read_text(encoding="utf-8")) == {"source": source}
        assert (report_dir / f"{stem}_report.txt").read_text(encoding="utf-8") == source


def test_run_ocr_pipeline_writes_requested_output_stem(tmp_path):
    class Document:
        def export_to_dict(self):
            return {"ok": True}

        def export_to_markdown(self, image_mode):
            return "converted"

    class Converter:
        def convert(self, path):
            return type("Result", (), {"document": Document()})()

    source = tmp_path / "report.pdf"
    source.write_bytes(b"fixture")
    markdown, data = run_ocr_pipeline(source, Converter(), tmp_path / "output", output_stem="report_pdf")

    assert markdown == "converted"
    assert data == {"ok": True}
    assert (tmp_path / "output" / "report_pdf.md").read_text(encoding="utf-8") == "converted"


def test_empty_input_skips_converter_and_keeps_output_directory(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    def unexpected_converter():
        raise AssertionError("Converter should not initialize when there are no supported files")

    monkeypatch.setattr("ocr_llm_pipeline.build_converter", unexpected_converter)
    output_dir = tmp_path / "output"
    process_documents(input_dir, output_dir)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_output_stem_cannot_escape_output_directory(tmp_path):
    class UnexpectedConverter:
        def convert(self, path):
            raise AssertionError("Output name must be validated before conversion")

    with pytest.raises(ValueError, match="output_stem"):
        run_ocr_pipeline(tmp_path / "input.pdf", UnexpectedConverter(), tmp_path / "output", output_stem="../outside")
    assert not (tmp_path / "output").exists()
