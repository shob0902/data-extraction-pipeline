## Overview

`data_extraction.py` is a Python script for extracting structured data from scanned documents using OCR text and page images in a multi-modal workflow.

It converts a PDF into page images, loads AWS Textract-style OCR output, and sends the combined content to an OpenAI responses model for JSON extraction. The script also includes a Gradio UI for running extraction and asking questions against the extracted document JSON.

## Features

- Convert PDF pages to JPEG images using `PyMuPDF` and `pdf2image`
- Load Textract JSON text blocks
- Call OpenAI Responses API with a structured system prompt for document extraction
- Batch pages in groups of three
- Log OpenAI usage to CSV
- Interactive Gradio app for extraction and question answering

## Requirements

- Python 3.10+ (recommended)
- `openai`
- `pydantic`
- `fitz` (PyMuPDF)
- `pdf2image`
- `tiktoken`
- `gradio`
- `PyPDF2` or similar PDF handling is not required directly
- Poppler utilities installed for `pdf2image`

## Installation

Install Python dependencies:

```bash
pip install openai pydantic pymupdf pdf2image gradio tiktoken
```

Install Poppler:

- On Windows: install Poppler for Windows and add its `bin` folder to `PATH`
- On Linux: `sudo apt-get update && sudo apt-get install -y poppler-utils`

## Configuration

1. Open `data_extraction.py`.
2. Replace `OPENAI_API_KEY = "your_openai_api_key_here"` with your actual OpenAI API key.
3. Optionally update `USAGE_LOG_PATH` if you want logs written somewhere else.

## Usage

Run the script from the folder containing `data_extraction.py`:

```bash
python data_extraction.py
```

This launches a Gradio interface where you can:

- Upload a PDF file
- Upload a corresponding Textract JSON file
- Run extraction to generate structured JSON output
- Ask questions about the extracted document

## Expected Inputs

- `PDF`: The source document to extract, one or more pages
- `Textract JSON`: OCR output in a JSON format containing `expenseData.textract_info` blocks with `Text` fields

## Notes

- The script currently hardcodes the OpenAI API key in `data_extraction.py`; store it securely or modify the script to read from an environment variable.
- Batch size for OpenAI calls is controlled by `BATCH_SIZE = 3`.
- The script uses a detailed system prompt to enforce strict JSON-only extraction.

## File

- `data_extraction.py`: main extraction app and Gradio UI

## Troubleshooting

- If PDF conversion fails, verify Poppler is installed and available on your `PATH`.
- If OpenAI calls fail, verify the API key and network connectivity.
- If the Textract JSON format changes, update `load_textract_texts()` to match the new schema.
