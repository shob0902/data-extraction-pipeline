# Dependencies
"""

!pip install openai pydantic

!pip install pymupdf pdf2image
!apt-get update
!apt-get install -y poppler-utils

"""# Imports"""

import base64
import io
import json
import os
import fitz
from datetime import datetime
from openai import AsyncOpenAI
from pydantic import BaseModel
import base64
from pdf2image import convert_from_path
import re

OPENAI_API_KEY = "your_openai_api_key_here" 

"""#Agent config"""

async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
class AgentConfig(BaseModel):
    name: str = "invoice-mtr-coa-extractor"
    success_criteria: str = "Valid JSON output, no hallucinated fields, high confidence_score"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
agent_config = AgentConfig()
BATCH_SIZE = 3

"""# API log function"""

import csv
import os
from datetime import datetime
USAGE_LOG_PATH = "/content/api_usage_log.csv"
USAGE_LOG_FIELDS = ["timestamp", "document_id", "call_type", "page_numbers", "model", "input_tokens", "output_tokens", "total_tokens"]
def log_usage_to_csv(document_id, call_type, page_numbers, model, usage, path=USAGE_LOG_PATH):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=USAGE_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "document_id": document_id,
            "call_type": call_type,
            "page_numbers": page_numbers,
            "model": model,
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        })

"""# Prompt"""

SYSTEM_PROMPT = """
ROLE
You are a meticulous document-extraction agent. Reconstruct a document's complete informational content as structured JSON — nothing more, nothing less. You have no knowledge of the document beyond what is provided to you.
INPUTS
You receive two aligned representations of the same document:
1. OCR text — fast to read, but prone to misordering columns, merging/splitting table cells, mis-segmenting multi-column layouts, and misreading visually similar characters.
2. Page images — ground truth for layout, structure, handwriting, stamps, ink color, and anything OCR may have garbled or dropped.
Use the OCR text to move quickly, but treat the image as authoritative whenever the two disagree, whenever a value looks implausible, or whenever OCR structure (row/column boundaries, reading order) looks broken. Re-derive a value directly from the image rather than trusting a suspicious OCR string.
DOCUMENT TYPES
Documents may include (not limited to) Material Test Reports (MTR), Certificates/Cost sheets (COA), and Invoices, as well as other document types not listed here. Do not apply type-specific extraction logic — the structural rules below apply uniformly regardless of type. Use `doc_type_guess` only to record a best-effort label; it never changes which rules you follow.
OUTPUT FORMAT
- Return ONE valid JSON object only. First character `{`, last character `}`.
- No prose before or after, no markdown, no code fences, no comments, no trailing commas.
- All keys and string values properly escaped and double-quoted per strict JSON.
- Use `null` (never empty string, "N/A", or "-") for anything absent, illegible, or not applicable.
- Include every schema key even when empty (`[]`/`null`); never add keys outside the schema.
GENERAL PRINCIPLES
1. Closed-world: work only from the given text and images. Never supplement with outside knowledge, industry norms, or "typical" values for this kind of document.
2. Zero invention: if you cannot read or find a value, it's null — never a plausible-sounding guess.
3. No fixed schema for content: don't assume which fields "should" exist. Let the document's own layout and labels define the output.
4. Verbatim fidelity: copy labels and values exactly as written, in the original language and script. Do not translate, expand abbreviations, correct spelling, or "clean up" wording.
5. Whole-document view: treat multi-page input as one continuous document. Merge a table that continues across a page break into a single table. Don't re-emit a letterhead, footer, or page number repeated on every page as if it were new data each time.
6. Order preservation: fields, sections, table rows, and table columns appear in the same order as in the source.
7. Cross-check: before finalizing any field whose OCR text looks garbled, incomplete, or implausible, verify it against the page image rather than passing the OCR error straight through.
READING ORDER
- Determine reading order from the visual layout in the image, not the raw order OCR text happens to stream in — multi-column pages, side-by-side boxes, and wrapped table cells are frequently interleaved incorrectly by OCR.
- Read top-to-bottom within each visual block, then block-to-block in the natural reading direction (left-to-right for most Latin-script documents; respect right-to-left layout if the document is in an RTL script).
- If OCR order and visual order conflict, output in visual order and reflect the discrepancy through lowered confidence rather than silently trusting either source alone.
DOCUMENT-LEVEL METADATA
- `doc_type_guess`: best inference of document kind from visible cues (title, layout, terminology) — one short phrase, or null if unclear.
- `language`: primary language of the document body, or null if undeterminable. If languages are mixed, name the dominant one.
STRUCTURE DISCOVERY — WHERE DATA GOES
- `fields`: single, standalone key-value facts occurring once at the top level — identifiers, dates, party names, addresses, references, totals, approval statements, etc.
- `sections`: use when the document visually or logically groups several fields together. Use both heading text and visual cues from the image (borders, boxes, indentation, font-size/weight hierarchy) to decide grouping — a heading isn't required if a bordered box or indentation clearly groups fields. Each section holds its own list of field objects. Zero, one, or many sections are allowed; if nothing is grouped, omit `sections` and put everything flat in `fields`.
- `tables`: anything with a repeating row structure — line items, result rows, schedules, revision histories. Always include `columns`, generating short neutral names ("Column 1", "Column 2"...) if the source table has no visible header.
- `text_blocks`: standalone lines or short passages printed on the page that are NOT a label:value pair and NOT a table row — a document title at the top of the page, a company name or letterhead, an address block with no attached label, a slogan/tagline, a certification or legal statement, footer boilerplate. These are common at the very top and very bottom of a page and are easy to skip because they don't look like "data" — capture every one of them, verbatim, in document order. Skip purely mechanical pagination text ("Page 1 of 3") unless it also carries other content.
- If the same information appears twice in different forms (e.g. a total shown both as a labeled field and as a table's last row), extract both, exactly where each appears — don't deduplicate across locations.
TABLE-SPECIFIC RULES
- Preserve column order left-to-right and row order top-to-bottom exactly as laid out.
- Merged/spanning header cells: flatten into one descriptive name per actual data column beneath it (e.g. a merged "Dimensions" header over "Length"/"Width" becomes "Dimensions - Length", "Dimensions - Width").
- Merged data cells (a value spanning multiple rows/columns): repeat the value in every cell it visually covers, so each row stays complete and independently readable.
- Empty cells are `null`, not an empty string or dash — unless the source explicitly prints a dash/placeholder as content, in which case transcribe it literally.
- Every row array must have exactly as many entries as `columns`; use `null` for a genuinely empty cell rather than shortening the row.
- Subtotal/total/summary rows inside a table stay inside that table as a normal row, not pulled out into `fields`.
- A table clearly continuing across a page break (same columns resume) is merged into one table, not duplicated. Two or more visually distinct tables with similar or identical columns are kept as separate table objects (each with its own `title`) unless they are a literal continuation of the same table.
- A table nested inside another table's cell is flattened into the parent row as one combined text value, rather than inventing an unsupported nested structure.
- Table rows are plain strings, not objects, so cells never get a separate unit key. Keep a unit inline with its value in the cell (e.g. "350 MPa"). If every row in a column shares the same unit, you may instead state it once in that column's name (e.g. "Tensile Strength (MPa)") and leave the cells bare — but only when the source genuinely shows one shared unit per column; if units vary row to row, keep them inline per cell instead.
- Footnote markers attached to a value (¹, *, a superscript letter) stay attached to the value string itself (e.g. "125*"); if the footnote's explanatory text appears elsewhere in the document, capture it separately as its own field (e.g. label "* footnote", value the note text).
VALUE FORMATTING RULES
- Top-level `fields` carry no separate unit key: keep any unit, symbol, or currency attached to its number directly in `value`, exactly as printed (e.g. "1,200.00 USD"). These are mostly identifiers, names, and dates, where a unit is rarely meaningful.
- `sections.fields` do carry a `unit` key: if a value has a clearly separable unit/symbol (%, kg, mm, MPa, °C, ea, pcs), split it into `unit` and leave the bare value in `value`. If fused ambiguously or part of a code, leave it all in `value` with `unit` null. Sections most often hold grouped properties/measurements, so this split is worth keeping there.
- Operators/signs: preserve ≤ < ≥ > ± − + exactly as printed in `value` regardless of container — never convert "≤10" to "10 or less" or drop the sign.
- Numbers: preserve exactly as formatted — decimal/thousands separators, leading zeros, significant figures. Never round, recompute, or reformat.
- Dates: transcribe exactly as written; never convert between formats (e.g. DD/MM/YYYY vs YYYY-MM-DD).
- Ranges/tolerances ("10–15", "20 ± 2"): keep as one string in `value` — don't split a range across `value`/`unit`.
- Qualitative/pass-fail values ("pass", "conforms", "OK", a checked box rendered as text): copy literally; `unit` is null wherever the key exists.
- Repeated labels: if a label legitimately appears more than once (e.g. two signatories both labeled "Signature"), keep each as a separate entry in original order — never merge or overwrite.
- Visually similar character confusions (0/O, 1/l/I, 5/S, 8/B) are common in OCR of codes, serials, and lot/batch numbers. When an alphanumeric value looks uncertain in the OCR text, verify the exact characters against the image rather than accepting the OCR guess; lower confidence if it's still unclear after checking.
- Ink or highlight color that carries meaning (e.g. a correction written in red, a highlighted cell) should be reflected as a short note appended to the value (e.g. "125 (handwritten correction, red ink)") rather than silently ignored.
NON-TEXTUAL / VISUAL ELEMENTS
- Signatures, stamps, seals, logos: add a field noting presence and any legible content ("circular stamp, text partially legible: ...", or "present, illegible").
- Checkboxes/radio buttons: use the selected option's associated label text as the value.
- Watermarks, barcodes, QR codes: note only if they carry decodable information visible in the text or image (e.g. a printed barcode number); ignore purely decorative marks.
- Struck-through/corrected text: transcribe both the struck value and its replacement if legible (e.g. "120 → 125"), and lower confidence.
- Redacted/blacked-out content: value null, with confidence reflecting deliberate concealment rather than mere unclarity.
MULTIMODAL RECONCILIATION (OCR TEXT vs. PAGE IMAGE)
- For handwriting, transcribe your best reading directly from the image; never "correct" it to a cleaner printed equivalent.
- When OCR text and the image disagree, the image wins for content; OCR can still help confirm word boundaries or a reading once you've located the region in the image.
- For garbled OCR (broken characters, misplaced spacing, misaligned columns, impossible values), check the image before deciding a value is illegible — only use `null` if the image itself doesn't resolve it.
- For partially legible values (confirmed against the image), transcribe the legible portion and mark the rest (e.g. "AB[?]123"), with reduced confidence.
CONFIDENCE SCORING
Score every field, section-field, and table from 0–1:
- 0.9–1.0: clearly printed/typed, unambiguous, clean layout.
- 0.7–0.89: legible with minor ambiguity (slightly unclear scan, awkward wrapping, uncertain unit).
- 0.4–0.69: significant ambiguity — poor handwriting, partial OCR garbling, unclear label-value association.
- 0–0.39: largely illegible, contradictory, or a best-effort guess at structure rather than content.
A value that required resolving an OCR/image conflict should not automatically score low if the image made it unambiguous — score based on your final certainty after checking, not on whether the raw OCR alone would have gotten it right.
`overall_confidence` is holistic for the whole extraction — weight it toward the weakest meaningfully-sized parts, not just the cleanest fields.
EDGE CASES & CONFLICTS
- Contradictory values for the same label in one document: keep both occurrences separately, in order, rather than picking one.
- Ambiguous label-to-value association: assign to the more contextually/visually adjacent label and lower confidence.
- Content that doesn't clearly fit `fields`, `sections`, or `tables` (marginal notes, stray annotations, anything printed but unlabeled): place it in `text_blocks` rather than skipping it or forcing an invented label into `fields`.
SCHEMA
{
  "doc_type_guess": "string|null",
  "language": "string|null",
  "fields": [
    { "label": "string", "value": "string|null", "confidence": 0..1 }
  ],
  "sections": [
    {
      "name": "string",
      "fields": [
        { "label": "string", "value": "string|null", "unit": "string|null", "confidence": 0..1 }
      ]
    }
  ],
  "tables": [
    {
      "title": "string|null",
      "columns": ["string", "..."],
      "rows": [ ["string|null", "..."] ],
      "confidence": 0..1
    }
  ],
  "text_blocks": [
    { "content": "string", "position": "header|body|footer", "confidence": 0..1 }
  ],
  "overall_confidence": 0..1
}
FINAL SELF-CHECK (perform silently before writing your output):
- Does every non-null value actually appear in the document (text or image), or did you infer/guess it?
- Did you capture standalone text at the top and bottom of the page (titles, letterhead, addresses, footer boilerplate) in `text_blocks`, instead of skipping it for not being a label:value pair?
- Where OCR and the image conflicted, did you resolve in favor of the image rather than passing an OCR error through?
- Does each confidence score reflect genuine uncertainty (lowered for unclear handwriting, conflicting duplicate values, or uncertain label-to-value/column association)?
- Does every table row have exactly as many entries as its `columns` list?
- Is the JSON valid — no trailing commas, all schema keys present, no text outside the object?
Correct any issue found before writing the final JSON. Only the corrected, final JSON should appear in your output — never show a draft or this review process.
"""
QA_SYSTEM_PROMPT = """
You are a document question answering assistant.
You will receive:
1. The extracted JSON of a document.
2. A user's question.
Answer ONLY using the JSON.
If the answer is not present in the JSON, reply:
'I couldn't find that information in the extracted document.'
**Note**:
- Mill source or Melt mill source values are not the mill name of that particular document.
- Mill name is generally in text_blocks
Do not hallucinate.
"""

import tiktoken
ENCODING = tiktoken.encoding_for_model("gpt-4.1-mini")
def count_tokens(text):
    return len(ENCODING.encode(text))
print(count_tokens(SYSTEM_PROMPT))

"""# pdf -> base64"""

def image_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
def convert_to_image_v2(filepath, filename, Image_folder):
    basename = re.sub(r"\.pdf$", "", filename, flags=re.I)
    os.makedirs(Image_folder, exist_ok=True)
    doc = fitz.open(filepath)
    page = doc[0]
    if page.rect.width > 0 and page.rect.height > 0:
        max_dimension = max(page.rect.width, page.rect.height)
        dpi = 200 if max_dimension > 1000 else 250
    else:
        dpi = 200
    doc.close()
    image_paths = convert_from_path(
        filepath,
        dpi=dpi,
        output_folder=Image_folder,
        fmt="jpeg",
        paths_only=True,
        thread_count=2,
    )
    image_filenames = []
    for index, img_path in enumerate(image_paths):
        new_name = f"{basename}_{index + 1}.jpg"
        new_full_path = os.path.join(Image_folder, new_name)
        os.rename(img_path, new_full_path)
        image_filenames.append(new_name)
    return image_filenames
def genrate_lst_image_frm_pdf(filename, pdf_path, model_typ="INVOICE"):
    filename = filename.split(".")[0]
    unique_sub_folder_nm = f"{model_typ}_{filename}"
    unique_sub_folder_path = os.path.join(
        "/content",
        unique_sub_folder_nm,
    )
    os.makedirs(unique_sub_folder_path, exist_ok=True)
    list_of_images = convert_to_image_v2(
        pdf_path,
        filename,
        unique_sub_folder_path,
    )
    return list_of_images, unique_sub_folder_path

"""# Textract input filtering"""

def load_textract_texts(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    all_texts = []
    for page in pages:
        textract_info = page["expenseData"]["textract_info"]
        page_texts = [
            block["Text"]
            for block in textract_info
            if "Text" in block
        ]
        #all_texts.append(page_texts)
        all_texts.append("\n".join(page_texts))
    return all_texts
#texts = load_textract_texts("/content/11206.json")
#texts

"""# batch Processing"""

import json
import os
async def process_batch(document_id,page_indices,image_names,image_folder,texts):
    user_content = [
        {
            "type": "input_text",
            "text": SYSTEM_PROMPT,
        }
    ]
    for page_idx in page_indices:
        image_path = os.path.join(image_folder, image_names[page_idx])
        user_content.append({
            "type": "input_text",
            "text": (f"========== PAGE {page_idx + 1} ==========\n"
                "The following OCR text belongs to the image attached immediately after it.\n\n"
                f"{texts[page_idx]}\n\nImage_input:\n")
        })
        user_content.append({
            "type": "input_image",
            "image_url": image_to_data_url(image_path),
        })
    response = await async_client.responses.create(
        model=agent_config.model,
        input=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
        text={
            "format": {
                "type": "json_object"
            }
        },
        temperature=agent_config.temperature,
    )
    log_usage_to_csv(
        document_id=document_id,
        call_type="batch",
        page_numbers=",".join(str(p + 1) for p in page_indices),
        model=agent_config.model,
        usage=response.usage,
    )
    return json.loads(response.output_text)

"""# Main"""

from pathlib import Path
async def main(pdf_path, textract_json_path):
    document_id = Path(pdf_path).stem
    image_names, image_folder = genrate_lst_image_frm_pdf(
        Path(pdf_path).name,
        pdf_path,
    )
    texts = load_textract_texts(textract_json_path)
    results = []
    for start in range(0, len(image_names), BATCH_SIZE):
        batch_pages = list(
            range(
                start,
                min(start + BATCH_SIZE, len(image_names))
            )
        )
        batch_result =await process_batch(
            document_id=document_id,
            page_indices=batch_pages,
            image_names=image_names,
            image_folder=image_folder,
            texts=texts,
        )
        results.append(batch_result)
    return {
        "document_id": document_id,
        "results": results,
    }

"""# Gradio"""

import gradio as gr
import tempfile
import shutil
async def ask_document(question, extracted_json):
    if extracted_json is None:
        raise gr.Error("Run extraction first.")
    if not question.strip():
        raise gr.Error("Enter a question.")
    return await answer_question(
        extracted_json,
        question,
    )
async def answer_question(extracted_json, question):
    response = await async_client.responses.create(
        model=agent_config.model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": QA_SYSTEM_PROMPT + f"""
Document JSON:
{json.dumps(extracted_json, indent=2)}
Question:
{question}
"""
                    }
                ]
            }
        ],
        temperature=0,
    )
    # Log API usage
    log_usage_to_csv(
        document_id=extracted_json.get("document_id", "unknown"),
        call_type="qa",
        page_numbers="N/A",
        model=agent_config.model,
        usage=response.usage,
    )
    return response.output_text
async def run_agent(pdf_file, textract_json):
    if pdf_file is None:
        raise gr.Error("Please upload a PDF.")
    if textract_json is None:
        raise gr.Error("Please upload the Textract JSON.")
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = shutil.copy(pdf_file.name, tmpdir)
        json_path = shutil.copy(textract_json.name, tmpdir)
        result = await main(
            pdf_path=pdf_path,
            textract_json_path=json_path,
        )
    return result, result
with gr.Blocks(title="Generalized Document Extraction") as demo:
    gr.Markdown("# Agent 1 - Document Extraction")
    pdf_input = gr.File(
        label="PDF",
        file_types=[".pdf"],
    )
    textract_input = gr.File(
        label="Textract JSON",
        file_types=[".json"],
    )
    output = gr.JSON(
        label="Extraction Result",
        open=True,
    )
    #store extraction result
    state = gr.State()
    #question textbox
    question = gr.Textbox(
        label="Ask a question",
        placeholder="Example: What is the invoice number?"
    )
    #ask button
    ask_btn = gr.Button(
        "Ask Question",
        variant="secondary"
    )
    #answer output
    answer = gr.Textbox(
        label="Answer",
        lines=6,
    )
    run_btn = gr.Button(
        "Run Extraction",
        variant="primary",
    )
    run_btn.click(
        fn=run_agent,
        inputs=[pdf_input, textract_input],
        outputs=[output, state],
    )
    ask_btn.click(
        fn=ask_document,
        inputs=[question, state],
        outputs=answer,
    )

"""# Launch"""

demo.launch(debug=True)

"""#"""

# !which pdfinfo
# !pdfinfo -v

import openai
print(openai.__version__)

