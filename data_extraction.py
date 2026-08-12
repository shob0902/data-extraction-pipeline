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

"""
QA_SYSTEM_PROMPT = """

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

