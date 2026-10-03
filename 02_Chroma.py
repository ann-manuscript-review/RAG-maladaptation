import json
import re
from pathlib import Path

import requests
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 00. Configuration
UPSTAGE_API_KEY = "*********************"
OPENAI_API_KEY = "**********************"

PDF_DIR = Path(r"C:\path\to\articles")
OUT_DIR = Path(r"C:\path\to\processed_articles")
INDEX_DIR = Path(r"C:\path\to\ChromaDB")

LLM_MODEL = "gpt-5-mini-2025-08-07"
EMBED_MODEL = "BAAI/bge-m3"
UPSTAGE_URL = "https://api.upstage.ai/v1/document-digitization"
CROSSREF_URL = "https://api.crossref.org/works"

SUBDIRS = {name: OUT_DIR / name for name in ("html", "md", "meta")}
for d in SUBDIRS.values():
    d.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=OPENAI_API_KEY)

RE_DOI = re.compile(r"(10\.\s*\d[\d\s]{3,8}/[\w.\-()\s/;:+]+)", re.I)

def ask_llm(prompt: str, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        reasoning_effort="medium",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return resp.choices[0].message.content

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

# 01. PDF parsing
def parse_with_upstage(pdf_path: Path) -> str:
    payload = {
        "model": "document-parse-260930",
        "ocr": "force",
        "output_formats": ["html"],
        "merge_multipage_tables": True,
        "chart_recognition": True,
    }
    with open(pdf_path, "rb") as f:
        res = requests.post(
            UPSTAGE_URL,
            headers={"Authorization": f"Bearer {UPSTAGE_API_KEY}"},
            files={"document": f},
            data=payload,
        )
    res.raise_for_status()
    return res.json()["content"]["html"]

# 02. Bibliographic metadata
def extract_doi(text: str) -> str:
    m = RE_DOI.search(text)
    if not m:
        return ""

    print("DOI match:", repr(m.group(0)))
    doi = m.group(1)
    doi = re.sub(r"<.*$", "", doi)
    doi = re.sub(r"[\s]", "", doi)
    print("Extracted DOI:", repr(doi))
    return doi

def fetch_crossref_metadata(doi: str) -> dict:
    r = requests.get(f"{CROSSREF_URL}/{doi}", timeout=10)
    if r.status_code != 200:
        return {}

    item = r.json().get("message", {})
    first = lambda key: (item.get(key) or [""])[0]
    authors = [
        f"{a.get('given', '')} {a.get('family', '')}".strip()
        for a in item.get("author", [])
    ]
    return {
        "title": first("title"),
        "authors": ", ".join(a for a in authors if a),
        "year": str(item.get("issued", {}).get("date-parts", [[None]])[0][0]),
        "journal": first("container-title"),
        "doi": doi,
    }

def extract_meta_with_llm(text: str) -> dict:
    prompt = f"""
    Extract bibliographic metadata from the following academic article text.
    Return strict JSON with keys: title, authors, year, journal.

    Text:
    {text[:15000]}
    """
    return json.loads(ask_llm(prompt, json_mode=True))

def search_crossref_by_meta(title: str, authors: str, year: str) -> str:
    try:
        first_author = authors.split(",")[0] if authors else ""
        r = requests.get(
            CROSSREF_URL,
            params={"query.title": title, "query.author": first_author, "rows": 3},
            timeout=10,
        )
        if r.status_code != 200:
            return ""
        for it in r.json().get("message", {}).get("items", []):
            cand_title = (it.get("title") or [""])[0].lower()
            if title.lower()[:40] in cand_title:            
                return it.get("DOI", "")
    except Exception:
        pass
    return ""

def extract_metadata_pipeline(text: str) -> dict:
    doi = extract_doi(text)
    if doi:
        meta = fetch_crossref_metadata(doi)
        if meta:
            return meta

    meta = extract_meta_with_llm(text)
    doi = search_crossref_by_meta(
        meta.get("title", ""), meta.get("authors", ""), meta.get("year", "")
    )
    if doi:
        meta["doi"] = doi
    return meta

# 03. Main body text cleaning
CLEAN_PROMPT = """
# The following is raw text extracted from a PDF-formatted academic article.

# Your task is to extract and clean only the *main body text* from an academic article in plain text format.
# The goal is to remove all non-body elements while keeping the sentences of the main text exactly as written.

# Rules
## 1. Do not summarize, rephrase, or paraphrase; Keep the sentences exactly as in the original text.
## 2. Keep only the main body:
### Start after the "Introduction" or equivalent main section heading.
### If no explicit "Introduction" or similar heading exists, begin from the first narrative or analytical paragraph that follows the Abstract or Keywords
### Stop before any of the back matter sections as follows: References, Bibliography, Appendix, Acknowledgements, Author Contributions, Supplementary Information, or similar sections.
## 3. Remove these elements entirely:
### Tables, figures, graphs, and their captions (including text blocks beginning with "Table", "Figure", "Fig.", "Tab.", "Graph").
### Headers, footers, watermarks, footnotes, and page numbers.
### All citation references such as (Smith, 2020), [1], [12,13], etc.
## 4. Clean up formatting:
### Normalize spacing, remove excessive line breaks.
### Structure the text with Markdown headings (##, ###, ####: Section Title, and paragraphs).

# Output only the cleaned Markdown text.

Text: 
{text}
"""

def clean_with_gpt(text: str) -> str:
    return ask_llm(CLEAN_PROMPT.format(text=text[:200000]))

# 04. Metadata formatting
def flatten_meta(meta: dict) -> dict:
    return {
        k: ", ".join(map(str, v)) if isinstance(v, list) else v
        for k, v in meta.items()
    }

# 05. Main
def process_pdf(pdf_path: Path, db: Chroma) -> None:
    stem = pdf_path.stem

    html_text = parse_with_upstage(pdf_path)
    write_text(SUBDIRS["html"] / f"{stem}.html", html_text)

    meta = extract_metadata_pipeline(html_text)
    write_text(SUBDIRS["meta"] / f"{stem}.json",
               json.dumps(meta, ensure_ascii=False, indent=2))

    md_text = clean_with_gpt(html_text)
    write_text(SUBDIRS["md"] / f"{stem}.md", md_text)

    doc = Document(
        page_content=md_text,
        metadata={"source": str(pdf_path), **flatten_meta(meta)},
    )
    db.add_documents([doc])

def main() -> None:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=str(INDEX_DIR), embedding_function=emb)

    for pdf_path in PDF_DIR.glob("*.pdf"):
        print(f"Processing: {pdf_path.name}")
        try:
            process_pdf(pdf_path, db)
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")

    db.persist()

if __name__ == "__main__":
    main()
