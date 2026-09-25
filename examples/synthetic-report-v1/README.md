# Synthetic report fixture v1

`generate.py` creates a fixed one-page text PDF using only Python's standard library. All names and financial values are synthetic. The input generator is versioned; the generated PDF contains no timestamps or random identifiers. This is a native-text PDF, not a scanned-document OCR benchmark.

`reference.md` is a **human-authored target representation**, not captured pipeline output. Exact Markdown formatting can vary; assess text preservation and table structure separately.

From the repository root:

```bash
python examples/synthetic-report-v1/generate.py
python src/ocr_llm_pipeline.py \
  --input-dir examples/synthetic-report-v1/input \
  --markdown-dir md_results/synthetic-report-v1
```

The generation step needs no packages, credentials, or network. The pipeline step requires `requirements.txt` and may download model assets on first use. It does not request an Ollama report unless `--run-llm` is supplied.

## Review the output

| Check | Target | Failure to record |
| --- | --- | --- |
| Text | Title, three region names, six revenue values, closing sentence | Missing or altered characters/numbers |
| Structure | Region + Q1 + Q2 columns, one row per region | Rows flattened into a paragraph; wrong cell associations |
| Provenance | Exact generated PDF and source revision retained | Comparing output to a different local input |
| Optional report | Statements supported by extracted values | A fluent summary concealing extraction errors |

A [captured local run from 25 September 2026](CAPTURED_RUN_2026-09-25.md) archives the raw Markdown and JSON for this exact generated PDF. It completed but flattened the table into text; the JSON contains no table object. Use that output as an observed failure case, not as a target or quality benchmark. No Ollama call was made.

For a reportable run, retain the source commit, Python and package versions, PDF SHA-256, command, raw Markdown/JSON, hardware, and wall time. Keep generated outputs outside Git unless deliberately reviewed as public synthetic evidence. Do not report accuracy or general speed from this one document.
