# 01. Settings 
# ---------------------------------------------------------
import os
import re
from pathlib import Path
import requests
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from PyPDF2 import PdfReader, PdfWriter

# 01.1. Path
BASE = Path(r"C:\path\to\your\project")
PDF_DIR = BASE / "CCAP_Action_Plan"  # Folder for original planning documents (PDF)
OUT_DIR = BASE / "Output"            # Folder for saving results
CHROMA_DIR = BASE / "ChromaDB"       # Folder for Chroma used for RAG pipeline
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 01.2. API key
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "*****************") # Upstage API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "*******************") # OpenAI API Key
LLM_MODEL = "gpt-5-mini-2025-08-07"

# 01.3. Embedding model and DB settings 
EMB_MODEL = "BAAI/bge-m3"
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)

# 01.4. Re-ranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)

# 02. Passer API Call (Upstage Document Parse)
# ---------------------------------------------------------
def parse_with_upstage(pdf_path: Path) -> str:
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}

    with open(pdf_path, "rb") as f:
        files = {"document": f}
        payload = {
            "model": "document-parse", "ocr": "auto", "output_formats": ["html"],
            "merge_multipage_tables": True, "chart_recognition": True
        }

        res = requests.post(url, headers=headers, files=files, data=payload)

    if res.status_code != 200:
        raise RuntimeError(
            f"Passer API error ({res.status_code})\n"
            f"Response: {res.text[:500]}..."
        )

    data = res.json()
    html = data.get("content", {}).get("html", "")
    if not html.strip():
        raise ValueError("Parsed HTML is empty.")

    out_path = OUT_DIR / f"{pdf_path.stem}_parsed.html"
    out_path.write_text(html, encoding="utf-8")

    return html

# 03. Large-PDF Splitting + Parsing + HTML Merging
# ---------------------------------------------------------
def list_pdfs(max_n: int = 10):  # Return list of PDF files in the PDF directory.
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files: {PDF_DIR}")
    return pdf_files

def pick_pdf_by_fragment(fragment: str) -> Path:  # Select a PDF file by matching part of its filename.
    frag = fragment.lower()
    matches = [p for p in PDF_DIR.glob("*.pdf") if frag in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No PDF matching fragment: '{fragment}'")
    matches.sort(key=lambda p: len(p.name)) 
    return matches[0]

def split_pdf(input_path: Path, output_dir: Path, max_pages: int = 90):  # Split a PDF into multiple parts with max_pages per part.
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    parts = (total_pages // max_pages) + (1 if total_pages % max_pages else 0)
    output_paths = []

    for i in range(parts):
        writer = PdfWriter()
        start, end = i * max_pages, min((i + 1) * max_pages, total_pages)
        for j in range(start, end):
            writer.add_page(reader.pages[j])

        out_path = OUT_DIR / f"{input_path.stem}_part{i+1}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        output_paths.append(out_path)

    return output_paths

def parse_large_pdf_with_upstage(pdf_path: Path, max_pages: int = 90) -> str:  # Parsing each segment + merging (HTML). 
    split_paths = split_pdf(pdf_path, OUT_DIR, max_pages=max_pages)
    merged_html = ""

    for i, part_path in enumerate(split_paths, start=1):
        try:
            html_chunk = parse_with_upstage(part_path)
            merged_html += f"\n<!-- PART {i} START -->\n" + html_chunk + f"\n<!-- PART {i} END -->\n"
        except Exception:  
            continue

    merged_path = OUT_DIR / f"{pdf_path.stem}_merged.html"
    merged_path.write_text(merged_html, encoding="utf-8")
    return merged_html

def extract_section_by_fixed_keywords(pdf_path: Path, min_page_threshold: int = 10, max_section_pages: int = 90) -> list[Path]:  # Extract a section of a PDF bounded by fixed Korean keywords, then split into chunks if the section exceeds max_section_pages.
    reader = PdfReader(pdf_path)
    start_page, end_page = None, None

    start_kw = "(E.g.)부문별세부시행계획"  # Starting keyword; enter the wording exactly as written in the document
    end_kw = "(E.g.)계획의집행및관리"      # Ending keyword; enter the wording exactly as written in the document

    # 03.1. Locate start and end pages
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_clean = re.sub(r"\s+", "", text)

        if start_page is None and i > min_page_threshold and start_kw in text_clean:
            start_page = i
        elif start_page is not None and end_kw in text_clean:
            end_page = i
            break

    if start_page is None:
        raise ValueError(f"Start keyword '{start_kw}' not found after page {min_page_threshold}.")
    if end_page is None:
        end_page = len(reader.pages)

    total_pages = end_page - start_page

    # 03.2. Split extracted section if required
    section_parts = []
    for idx in range(0, total_pages, max_section_pages):
        writer = PdfWriter()
        part_start = start_page + idx
        part_end = min(start_page + idx + max_section_pages, end_page)
        
        for j in range(part_start, part_end):
            writer.add_page(reader.pages[j])

        part_path = OUT_DIR / f"{pdf_path.stem}_section_part{len(section_parts)+1}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)

        section_parts.append(part_path)

    return section_parts

# 03.3. Pipeline execution
files = list_pdfs()
target_pdf = pick_pdf_by_fragment("순천시")  # Select the plan, matching its name.
section_parts = extract_section_by_fixed_keywords(target_pdf, min_page_threshold=150, max_section_pages=90)  # 'min_page_threshold=150': start keyword search begins after page 150 / 'max_section_pages=90': split extracted section into chunks of up to 90 pages

merged_html = ""  
for i, section_pdf in enumerate(section_parts, start=1):
    html_chunk = parse_with_upstage(section_pdf)
    merged_html += f"\n<!-- SECTION {i} START -->\n" + html_chunk + f"\n<!-- SECTION {i} END -->\n"

merged_path = OUT_DIR / f"{target_pdf.stem}_section_merged.html"
merged_path.write_text(merged_html, encoding="utf-8")

# 04. Information Extraction Stage
# ---------------------------------------------------------
def ask_llm_on_document(html_path: Path, model: str = LLM_MODEL) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    document_text = html_path.read_text(encoding="utf-8")

    # 04.1. Prompt
    user_prompt = f"""
        # Extract the text exactly as written in the document. Do not paraphrase, rewrite, or modify wording.
        # Do not infer, assume, or supplement any information that is not explicitly stated in the document.
        # Exclude all labels, numbering, bracketed codes, and formatting markers from outputs; return only the descriptive text.
        
        # Objective/Action classification rules:       
        ## Only the sections in the document’s tables or main text that are explicitly marked as “추진전략” should be identified as {{objective}}.
        ## Only the sections in the document’s tables or main text that are explicitly marked as “실천과제” should be identified as {{action}}.
        ## Only structural labels (e.g., 추진전략, 실천과제) determine classification; Do not classify objectives and actions based on semantic meaning, wording style, and phrasing.
        
        # Maladaptation definition: 
        ## Maladaptation arises from unintended trade-offs created by implementing an action to achieve its objective—such as harms imposed on other policy goals, social groups, or spatial areas.
        ## Do not classify background problems, general negative conditions, or implementation challenges (e.g., costs, burdens, resource shortages) as maladaptation.
        ## Do not infer maladaptation unless explicitly stated; if no maladaptation is explicitly mentioned for an action, output “(Missing)”.
        
        # Extract maladaptation risks only for each {{action}} in relation to its corresponding {{objective}}.
        # Attach maladaptation output under each action, but treat maladaptation as occurring at the objective–action pair level.
        
        Output Format (must follow this structure strictly):
        # Objective: {{objective}}
        ## Action: {{action}}
        ## Maladaptation risks: ...
        (Repeat this block for all objectives)
        
        === DOCUMENT START ===
        {document_text}
        === DOCUMENT END ===
    """

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": user_prompt}],
    )

    final_text = resp.output_text.strip() if hasattr(resp, "output_text") else ""
    clean_stem = html_path.stem.replace("_section_merged", "")
    out_path = OUT_DIR / f"LLM_Extract_{clean_stem}.txt"
    out_path.write_text(final_text, encoding="utf-8")    

    return final_text

html_path = OUT_DIR / f"{target_pdf.stem}_section_merged.html"
ask_llm_on_document(html_path)

# 05. Inference stage: Objective-action pair inference
# ---------------------------------------------------------
def infer_missing_impacts(extracted_txt_path: Path, model: str = LLM_MODEL, top_k: int = 5):
    client = OpenAI(api_key=OPENAI_API_KEY)
    extracted_text = extracted_txt_path.read_text(encoding="utf-8")

    # 05.1. Seperate objective and action sections
    sections = re.split(r"(?=^# Objective:)", extracted_text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    total_results = []

    for sidx, sec in enumerate(sections, start=1):

        # 05.1.1. Objective
        obj_match = re.search(r"^# Objective:\s*(.+)", sec, flags=re.MULTILINE)
        if not obj_match:
            continue
        objective = obj_match.group(1).strip()

        # 05.1.2. Action and maladaptation
        pairs = re.findall(
            r"## Action:\s*(.+?)\s*## Maladaptation risks:\s*(.+?)(?=^## Action:|$)",
            sec,
            flags=re.MULTILINE | re.DOTALL,
        )

        if not pairs:
            continue

        # 05.1.3. Collect only missing actions
        missing_actions = []
        for action, risk in pairs:
            action = action.strip()
            risk = risk.strip()
            if risk == "(Missing)":
                missing_actions.append(action)

        if not missing_actions:
            continue

        # 05.2. Inference for each missing action
        for aidx, action in enumerate(missing_actions, start=1):

            # 05.2.1. Query for retrieval (Objective–Measure pair) 
            query = f"Maladaptation from implementing '{objective}' via measure '{action}'."

            # 05.2.2. Retrieval
            docs = db.similarity_search(query, k=top_k)

            # 05.2.3. Re-ranking
            scores = [reranker.compute_score([query, d.page_content]) for d in docs]
            reranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in reranked_docs[:3]]

            # 05.2.4. Build evidence text
            if top_docs:
                context_chunks = []
                for i, d in enumerate(top_docs):
                    src = d.metadata.get("source", "unknown")
                    title = d.metadata.get("title", "")
                    context_chunks.append(
                        f"[{i+1}] {src if src else title}:\n{d.page_content.strip()}"
                    )
                context_text = "\n\n".join(context_chunks)
            else:
                context_text = "(No evidence)"

            # 05.2.5. Prompt
            prompt = f"""
                     # Your task is to infer maladaptation that may arise when achieving the given {objective} through its {action}.

                     # Maladaptation definition: 
                     ## Maladaptation arises from unintended trade-offs created by implementing an action to achieve its objective—such as harms imposed on other policy goals, social groups, or spatial areas.
                     ## Do not classify background problems, general negative conditions, or implementation challenges (e.g., costs, burdens, resource shortages) as maladaptation.
                 
                     # Using only the contextual evidence provided below, infer maladaptation risks for each objective–action pair.
                     # If no evidence supports a maladaptation, write: "(No evidence-based maladaptation found)"
                 
                     ---
                     Objective: {objective}
                    
                     Actions: {action}

                     Contextual Evidence: {context_text}
                     ---
                 
                     # Instructions:
                     ## 1. Write ONE concise paragraph describing an evidence-supported maladaptation.
                     ## 2. End with an inline citation: (Evidence: Author, Year).
                     ## 3. If no evidence supports a maladaptation risk, output only: "(No evidence-based maladaptation found)"
                     ## 4. Respond in English.
                        
                     Output format:
                     # Inferred risk for: {objective} – {action}
                     [Paragraph OR (No evidence-based maladaptation found)]
                     """

            # 05.2.6. Invoke LLM
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
            )

            inference = resp.output_text.strip() if hasattr(resp, "output_text") else "(Inference failed)"
            total_results.append(inference + "\n")

    # 05.3. Save inference results
    if not total_results:
        return None

    final_output = "\n\n".join(total_results)

    clean_stem = extracted_txt_path.stem.replace("_SINGLE", "").replace("_section_merged", "").replace("LLM_Extract_", "")
    out_path = OUT_DIR / f"LLM_Inferred_{clean_stem}.txt"
    out_path.write_text(final_output, encoding="utf-8")

    return final_output

# 05.4. Run inference
extracted_txt_path = OUT_DIR / f"LLM_Extract_{html_path.stem.replace('_section_merged','')}.txt"
infer_missing_impacts(extracted_txt_path)

