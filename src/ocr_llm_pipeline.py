"""OCR/LLM pipeline based on the original Colab notebook `week2.ipynb`.

Pipeline:
1. preprocess images with OpenCV;
2. convert PDF/images to Markdown and JSON via Docling + RapidOCR;
3. optionally generate a short Russian analytical report via Ollama Cloud.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from collections import Counter
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling_core.types.doc import ImageRefMode


ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
DEFAULT_MODEL = "gemma4:31b-cloud"


def get_ollama_api_key() -> str:
    """Return Ollama API key from environment or Google Colab secret storage."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if api_key:
        return api_key

    try:
        from google.colab import userdata  # type: ignore

        api_key = userdata.get("ocr_olama")
        if api_key:
            return api_key
    except Exception:
        pass

    raise ValueError(
        "Ollama API key is not configured. Set OLLAMA_API_KEY or add Colab secret 'ocr_olama'."
    )


def preprocess_image(image_path: str | Path) -> str:
    """Sharpen and binarize an image before OCR; returns temporary file path."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    threshold = cv2.adaptiveThreshold(
        sharpen,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        15,
    )

    fd, output_name = tempfile.mkstemp(prefix="ocr_preprocessed_", suffix=".png")
    os.close(fd)
    output_path = Path(output_name)
    try:
        if not cv2.imwrite(str(output_path), threshold):
            raise OSError(f"Could not write preprocessed image: {output_path}")
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return str(output_path)


def build_converter() -> DocumentConverter:
    """Create Docling converter with OCR, formula enrichment and picture extraction."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_picture_images = True
    pipeline_options.ocr_options = RapidOcrOptions()

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }
    return DocumentConverter(format_options=format_options)


def run_ocr_pipeline(
    file_path: str | Path,
    converter: DocumentConverter,
    output_dir: str | Path,
    assets: bool = False,
    output_stem: str | None = None,
) -> tuple[str, dict]:
    """Convert a document/image to Markdown and Docling JSON dictionary."""
    path = Path(file_path)
    name = output_stem if output_stem is not None else path.stem
    if not name or Path(name).name != name or "\\" in name or name in {".", ".."}:
        raise ValueError("output_stem must be a single file name")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    is_image = path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    target_path = str(path)
    temp_file_to_clean: str | None = None

    if is_image:
        target_path = preprocess_image(path)
        temp_file_to_clean = target_path

    try:
        result = converter.convert(target_path)
    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            os.remove(temp_file_to_clean)

    doc_dict = result.document.export_to_dict()

    if assets:
        assets_dir = output_dir / f"{name}_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{name}.md"

        result.document.save_as_markdown(
            filename=md_path,
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=assets_dir,
        )
        markdown_text = md_path.read_text(encoding="utf-8")
    else:
        markdown_text = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        (output_dir / f"{name}.md").write_text(markdown_text, encoding="utf-8")

    return markdown_text, doc_dict


def run_llm_extraction(markdown_text: str, model_name: str = DEFAULT_MODEL) -> str:
    """Generate a short Russian analytical report from Markdown text via Ollama Cloud."""
    from llama_index.llms.openai_like import OpenAILike

    api_key = get_ollama_api_key()
    llm = OpenAILike(
        model=model_name,
        api_key=api_key,
        api_base="https://ollama.com/v1",
        is_chat_model=True,
        temperature=0.1,
        max_tokens=2048,
        timeout=360.0,
    )

    # The prompt below is deliberately written in Russian and asks for a
    # Russian-language report -- this pipeline targets Russian-language
    # document analysis, so the output language is a feature, not
    # untranslated text.
    prompt = f"""
Перед тобой структурированное представление документа в формате Markdown.
Твоя задача — внимательно проанализировать его и составить краткий аналитический отчет/резюме на русском языке (1-2 абзаца).

Сфокусируйся на:
- Основной цели документа / исследования.
- Ключевых выводах, результатах или метриках.

Важные требования:
- Пиши только связный текст отчета на русском языке.
- Не добавляй вводных фраз ("Вот ваш отчет:", "Конечно, я помогу" и т.д.).

Документ для анализа:
---
{markdown_text}
---
"""
    response = llm.complete(prompt)
    return response.text.strip()


def iter_input_files(input_dir: str | Path) -> Iterable[Path]:
    input_dir = Path(input_dir)
    return sorted(
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def _output_stems_for_files(files: list[Path]) -> dict[Path, str]:
    """Preserve unique stems and disambiguate names shared across input formats."""
    counts = Counter(path.stem.casefold() for path in files)
    reserved = {path.stem.casefold() for path in files if counts[path.stem.casefold()] == 1}
    used = reserved | {"pipeline-status"}
    output_stems: dict[Path, str] = {}

    for path in files:
        if counts[path.stem.casefold()] == 1 and path.stem.casefold() != "pipeline-status":
            output_stems[path] = path.stem
            continue

        base = f"{path.stem}_{path.suffix[1:].lower()}"
        candidate = base
        index = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{index}"
            index += 1
        output_stems[path] = candidate
        used.add(candidate.casefold())

    return output_stems


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _write_status(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".pipeline-status-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def process_documents(
    input_dir: str | Path,
    markdown_dir: str | Path,
    reports_dir: str | Path | None = None,
    assets: bool = False,
    run_llm: bool = False,
    model_name: str = DEFAULT_MODEL,
    resume: bool = False,
) -> None:
    if run_llm and not reports_dir:
        raise ValueError("reports_dir must be provided when run_llm=True")
    files = list(iter_input_files(input_dir))
    output_stems = _output_stems_for_files(files)
    markdown_dir = Path(markdown_dir)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(reports_dir) if reports_dir else None
    if reports_path:
        reports_path.mkdir(parents=True, exist_ok=True)
    print(f"Found files to process: {len(files)}")
    if not files:
        return

    status_path = markdown_dir / "pipeline-status.json"
    status = {"version": 1, "files": {}}
    if resume and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(status, dict) or status.get("version") != 1 or not isinstance(status.get("files"), dict):
            raise ValueError("Invalid pipeline status manifest")
    versions = {}
    packages = ["docling", "docling-core", "rapidocr-onnxruntime", "opencv-python-headless"]
    if run_llm:
        packages += ["llama-index-core", "llama-index-llms-openai-like", "openai"]
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    signature = {"pipeline_sha256": _digest(Path(__file__)), "packages": versions,
                 "assets": assets, "run_llm": run_llm, "model_name": model_name}

    def artifacts(stem: str) -> list[Path]:
        paths = [markdown_dir / f"{stem}.md", markdown_dir / f"{stem}.json"]
        if run_llm:
            paths.append(reports_path / f"{stem}_report.txt")
        if assets:
            paths.extend(sorted(p for p in (markdown_dir / f"{stem}_assets").rglob("*") if p.is_file()))
        return paths

    converter = None
    for source in files:
        stem = output_stems[source]
        identity = {**signature, "source_sha256": _digest(source)}
        previous = status["files"].get(source.name, {})
        expected = artifacts(stem)
        if (resume and previous.get("state") == "complete" and previous.get("identity") == identity
                and previous.get("artifacts") == {str(p.resolve()): _digest(p) for p in expected if p.is_file()}
                and all(p.is_file() for p in expected)):
            print(f"Skipping verified completed file: {source.name}")
            continue
        print(f"\nProcessing file: {source.name}")
        start_time = time.monotonic()
        record = {"state": "processing", "identity": identity}
        status["files"][source.name] = record
        _write_status(status_path, status)
        try:
            if converter is None:
                converter = build_converter()
            markdown_text, document = run_ocr_pipeline(source, converter, markdown_dir, assets, stem)
            (markdown_dir / f"{stem}.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
            if run_llm:
                report = run_llm_extraction(markdown_text, model_name=model_name)
                (reports_path / f"{stem}_report.txt").write_text(report, encoding="utf-8")
            if _digest(source) != identity["source_sha256"]:
                raise RuntimeError("Source changed during processing")
            record.update(state="complete", artifacts={str(p.resolve()): _digest(p) for p in artifacts(stem)})
        except Exception as error:
            record.update(state="failed", error_type=type(error).__name__)
            _write_status(status_path, status)
            raise
        _write_status(status_path, status)
        print(f"Completed file: {source.name} in {time.monotonic() - start_time:.2f} sec")
        print(f"Markdown/JSON saved to: {markdown_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR + LLM document processing pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory with PDF/images")
    parser.add_argument("--markdown-dir", required=True, help="Directory for Markdown/JSON OCR results")
    parser.add_argument("--reports-dir", default=None, help="Directory for LLM reports")
    parser.add_argument("--resume", action="store_true", help="Reuse only hash-verified completed outputs")
    parser.add_argument("--assets", action="store_true", help="Save extracted images as referenced assets")
    parser.add_argument("--run-llm", action="store_true", help="Generate LLM reports from Markdown")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama Cloud model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_documents(
        input_dir=args.input_dir,
        markdown_dir=args.markdown_dir,
        reports_dir=args.reports_dir,
        assets=args.assets,
        run_llm=args.run_llm,
        model_name=args.model,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
