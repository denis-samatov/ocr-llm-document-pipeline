# OCR + LLM Document Pipeline

A pipeline for processing PDFs and images:

1. OCR and document structuring via **Docling** + **RapidOCR**.
2. Image preprocessing via OpenCV.
3. Export results to Markdown and JSON.
4. Generate a concise analytical report via **Ollama Cloud** (`gemma4:31b-cloud`) through a LlamaIndex/OpenAI-like API.

The original notebook is saved at `notebooks/week2.ipynb`.

## Structure

```text
.
├── notebooks/
│   └── week2.ipynb
├── src/
│   └── ocr_llm_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment variables

To generate reports via Ollama Cloud, set an API key:

```bash
export OLLAMA_API_KEY="your_api_key"
```

In Google Colab, you can use the `ocr_olama` secret, as in the original notebook.

## Running

Example: process documents from the `ocr_samples` folder:

```bash
python src/ocr_llm_pipeline.py \
  --input-dir ./ocr_samples \
  --markdown-dir ./md_results \
  --reports-dir ./final_results \
  --assets \
  --run-llm
```

Supported formats: `.pdf`, `.jpg`, `.jpeg`, `.png`.

## Example

Given a one-page synthetic PDF containing:

```text
Quarterly Sales Report

Region      Q1 Revenue   Q2 Revenue
North       120,000      135,000
South       98,500       102,300
East        87,200       91,000

Prepared for internal review.
```

running just the OCR/Docling step —

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("sample_report.pdf")
print(result.document.export_to_markdown())
```

— produces this Markdown (captured from an actual run against the file above, not fabricated):

```markdown
Quarterly Sales Report

Region      Q1 Revenue   Q2 Revenue North       120,000      135,000 South       98,500       102,300 East        87,200       91,000

Prepared for internal review.
```

The `--run-llm` step then feeds this Markdown to Ollama Cloud and writes a short Russian-language analytical summary next to it (see [Environment variables](#environment-variables) for the API key it needs — that step isn't shown here since it requires a live Ollama Cloud credential).

## What gets generated

For each input document:

- a Markdown file with the recognized structure;
- a JSON file with the Docling object representation;
- a folder with images/artifacts, if `--assets` is enabled;
- a text analytical report, if `--run-llm` is enabled.

## Notes

- For the Colab version, use the original notebook `notebooks/week2.ipynb`.
- For local runs, use `src/ocr_llm_pipeline.py`.
- Do not commit real documents, OCR results, API keys, or temporary files to this repository.
