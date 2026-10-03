# Python codes and LLM prompts 

- This repository provides the Python code for the analysis (_RAG.py_) and evidence database construction (_Chroma.py_) to support reproducibility
- It also provides prompts, queries, output examples, and documentation on operational workflow.

## A. Technical setup and configuration 

- This section describes the system requirements, environment setup, directory structure, and model parameters required to reproduce the RAG pipeline used in this study.

### A.1. Required python libraries

- The pipeline uses the following key libraries: 

  - `openai` — LLM invocation (GPT-5-mini)
  - `requests` — Upstage Document Parse API call
  - `langchain_community.vectorstores.Chroma` — ChromaDB client
  - `langchain_community.embeddings.HuggingFaceEmbeddings` — bge-m3 embedding model
  - `FlagEmbedding` — bge-reranker-v2-m3 reranker
  - `PyPDF2` — PDF splitting and handling
  - `pathlib` — file management
  - `re` — regular expression parsing

- Recommended installation:

```bash
pip install openai langchain-community chromadb FlagEmbedding PyPDF2 requests
```

### A.2. Directory structure

- The following folders must exist befor running the pipeline:

```bash
BASE/
│── CCAP_Action_Plan/        # Original PDFs (local government plans)
│── ChromaDB/                # Pre-built vector store persisted by Chroma
│── Output/                  # LLM outputs: extracted text, HTML, inference results
│── RAG.py          # Main pipeline script
```

- In the script:

```python
BASE = Path(r"C:\path\to\your\project")
PDF_DIR = BASE / "CCAP_Action_Plan"
OUT_DIR = BASE / "Output"
CHROMA_DIR = BASE / "ChromaDB"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

### A.3. API keys and model configuration

- Set the following variables in the script to your own API keys before running the code:

  - `UPSTAGE_API_KEY`
  - `OPENAI_API_KEY`

- The pipeline uses the following models:

| Component | Model identifier |
|---|---|
| Document parsing | `document-parse-260930` (Upstage) |
| LLM | `gpt-5-mini-2025-08-07` (OpenAI) |
| Embedding | `BAAI/bge-m3` |
| Reranking | `BAAI/bge-reranker-v2-m3` |

- The same GPT-5-mini snapshot (`gpt-5-mini-2025-08-07`) and reasoning effort setting (`medium`) were used across all runs. 
- The code explicitly specifies this setting in both the information extraction and plausible maladaptation risk inference calls:

```python
reasoning={"effort": "medium"}
```

- No explicit output token limit (`max_output_tokens`) was set.
- For plausible maladaptation risk inference, the prompt instructed the model to return a single concise paragraph.

- The reranker is initialized with `use_fp16=False`.

### A.4. Evidence database construction

- The _Chroma.py_ script provides the code used to construct the Chroma evidence database.
- The bibliographic list of articles included in the corpus is provided in Table S10 of the Supplementary Materials.

- The Chroma database is not publicly distributed because it contains article body texts, including those from non-open-access publications.

- To construct the database locally, obtain authorized copies of the articles listed in Table S10 and configure the following directories:

| Variable | Description |
| --- | --- |
| `PDF_DIR` | Directory containing the article PDFs |
| `OUT_DIR` | Directory for extracted HTML, cleaned Markdown, and bibliographic metadata |
| `INDEX_DIR` | Directory for the persistent Chroma database |

- Example configuration:

```python
PDF_DIR = Path(r"C:\path\to\articles")
OUT_DIR = Path(r"C:\path\to\processed_articles")
INDEX_DIR = Path(r"C:\path\to\ChromaDB")
```

- The construction script parses the PDFs, extracts bibliographic metadata, cleans the article body texts, and stores the texts and their embeddings in Chroma.
- After database construction, set `CHROMA_DIR` in the main inference script (_RAG.py_) to the same directory as `INDEX_DIR`.

## B. Prompt and query
### B.1. Article body text cleaning

- The prompt below is provided in the `extract_meta_with_llm()` function in _Chroma.py_.

```text
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
### Tables, figures, graphs, and their captions (including text blocks beginning with "Table", "Figure", "Fig.", "Tab.". "Graph").
### Headers, footers, watermarks, footnotes, and page numbers.
### All citation references such as (Smith, 2020), [1], [12,13], etc.
## 4. Clean up formatting:
### Normalize spacing, remove excessive line breaks.
### Structure the text with Markdown headings (##, ###, ####: Section Title, and paragraphs).

# Output only the cleaned Markdown text.

Text:
{text}
```

### B.2. Information extraction 

- The prompt below is provided in the `ask_llm_on_document()` function in _RAG.py_.

```text
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

Output Structure Rules:
# Output each objective–action pair as a separate block.
# Each block must contain exactly one Objective, one Action, and one Maladaptation risks field.
# If multiple actions correspond to the same objective, create a separate block for each action and repeat the identical objective text in every block.
# Do not list multiple actions under a single Objective heading.
# Repeat this block for all objective–action pairs.
        
=== DOCUMENT START ===
{document_text}
=== DOCUMENT END ===
```
### B.3. Retrieval query 

- The prompt below is provided in the `infer_missing_impacts()` function in _RAG.py_.

```text
Maladaptation from implementing '{objective}' via measure '{action}'.
```

### B.4. Inference 

- The prompt below is provided in the `infer_missing_impacts()` function in _RAG.py_.

```text
# Your task is to infer maladaptation risks that may arise when achieving the given {objective} through its {action}.

# Maladaptation definition: 
## Maladaptation arises from unintended trade-offs created by implementing an action to achieve its objective—such as harms imposed on other policy goals, social groups, or spatial areas.
## Do not classify background problems, general negative conditions, or implementation challenges (e.g., costs, burdens, resource shortages) as maladaptation.
                 
# Using only the contextual evidence provided below, infer maladaptation risks for each objective–action pair.
# If no evidence supports a maladaptation risks, write: "(No evidence-based maladaptation found)"
                 
---
Objective: {objective}
                    
Actions: {action}

Contextual Evidence: {context_text}
---
                 
# Instructions:
## 1. Write ONE concise paragraph describing an evidence-supported maladaptation.
## 2. Cite supporting evidence using its exact ID in square brackets, e.g., [E1] or [E1] [E2].
## 3. Only cite evidence IDs provided in Contextual Evidence.
## 4. Do not write author-year citations yourself; these will be added programmatically.                     
## 5. If no evidence supports a maladaptation risk, output only: "(No evidence-based maladaptation found)"
## 6. Respond in English.
                        
Output Format (must follow this structure strictly):
# Inferred risk for: {objective} – {action}
[Paragraph OR (No evidence-based maladaptation found)]
```

## C. Output
### C.1. Output format instruction in prompt

- In this study, we instructed the model (prompt) to produce a hierarchical output structure as follows.
- Information extraction:

```text
Output Format (must follow this structure strictly):
# Objective: {objective}
## Action: {action}
## Maladaptation risks: ...
(Repeat this block for all objectives)
```

- Inference:

```text
Output Format (must follow this structure strictly):
# Inferred risk for: {objective} – {action}
[Paragraph OR (No evidence-based maladaptation found)]
```

### C.2. Output example: Geumjeong-gu, Busan

- The full results for the Geumjeong-gu, Busan case presented in Section Results are shown below.
- This serves as a sample illustrating that the same output structure is consistently produced across other local governments.
- In the "Extraction:" and "Inference:" examples below, the text in parentheses beginning with "ENG=" is not part of the original output but the authors’ English translation provided for readers who do not read Korean.
- Extraction: 

```text
# Objective: 기후변화로부터 구민 건강 보호 (ENG = Protecting public health from climate change)
## Action: 취약계층 중심 건강관리 강화 (ENG = Strengthening health management for vulnerable population groups)
## Maladaptation risks: (Missing)

# Objective: 기후변화로부터 구민 건강 보호 (ENG = Protecting public health from climate change)
## Action: 감염병 예방 및 신속 대응체계 강화 (ENG = Strengthening the prevention and rapid response system for infectious diseases)
## Maladaptation risks: (Missing)

# Objective: 구민 안전 확보 및 피해 최소화 (ENG = Ensuring public safety and minimizing climate-related risks)
## Action: 폭염으로부터 안전한 생활환경 조성 (ENG = Creating a heat-resilient community environment)
## Maladaptation risks: (Missing)

# Objective: 구민 안전 확보 및 피해 최소화 (ENG = Ensuring public safety and minimizing climate-related risks)
## Action: 체계적 풍수해 대응 관리 (ENG = Systematic management and response to storm and flood damage)
## Maladaptation risks: (Missing)

# Objective: 구민 안전 확보 및 피해 최소화 (ENG = Ensuring public safety and minimizing climate-related risks)
## Action: 미세먼지 대응 강화 (ENG = Enhancing particulate matter mitigation)
## Maladaptation risks: (Missing)

# Objective: 재해로부터 안전한 산림환경 구축 (ENG = Building a safe forest environment protected from disasters)
## Action: 산림종합방제 시스템 구축 (ENG = Establishing an integrated forest control system)
## Maladaptation risks: (Missing)

# Objective: 재해로부터 안전한 산림환경 구축 (ENG = Building a safe forest environment protected from disasters)
## Action: 기후변화 적응을 위한 산림 확대 (ENG = Expanding forest cover for climate change adaptation)
## Maladaptation risks: (Missing)

# Objective: 안정적 물이용 체계 확보 (ENG = Establishing a stable usage framework)
## Action: 안전한 물 공급 및 깨끗한 수자원 관리 (ENG = Ensuring a stable water supply and clean water resource management)
## Maladaptation risks: (Missing)

# Objective: 기후변화 대응 역량 강화 (ENG = Strengthening capacities for climate change response)
## Action: 저탄소 생활 실천 활성화 (ENG = Promoting the practice of low-carbon lifestyles)
## Maladaptation risks: (Missing)
```

- Inference: 

```text
# Inferred risk for: 기후변화로부터 구민 건강 보호 – 취약계층 중심 건강관리 강화 (ENG = Protecting public health from climate change - Strengthening health management for vulnerable population groups)
Targeting climate-health interventions narrowly at vulnerable groups (e.g., scaling clinical or short‑term measures such as increased provision of air‑conditioning or emergency services) risks maladaptation by substituting or transferring hazards and locking in path‑dependent, short‑term solutions: these actions can increase greenhouse gas emissions and energy costs (harming low‑income households), impose disproportionate burdens or worsen vulnerability in other groups or places, foreclose broader governance and preventive options (opportunity costs/path dependence), and shift risks across spatial and temporal scales if not integrated across sectors and planned with spatial scenario analysis (Evidence: Turner et al., 2023; Findlater et al., 2021; Chi et al., 2021).

# Inferred risk for: 기후변화로부터 구민 건강 보호 – 감염병 예방 및 신속 대응체계 강화 (ENG = Protecting public health from climate change - Strengthening the prevention and rapid response system for infectious diseases)
Strengthening infectious disease prevention and rapid‑response systems may produce maladaptation by creating unintended trade‑offs—for example, prioritizing techno‑centric surveillance, cold‑chains or emergency response can divert limited public‑health resources and staff from other essential services, concentrate benefits in well‑resourced urban populations while leaving rural or Indigenous communities more vulnerable (risk transfer/substitution), increase energy use or emissions and lock in path‑dependent solutions that raise long‑term vulnerability and inequities rather than reducing them (Evidence: Chi et al., 2021; Turner et al., 2023; Pourzand et al., 2023).

# Inferred risk for: 구민 안전 확보 및 피해 최소화 – 폭염으로부터 안전한 생활환경 조성 (ENG = Ensuring public safety and minimizing climate-related risks - Creating a heat-resilient community environment)
Cooling-focused interventions (e.g., widespread mechanical air conditioning, emergency cooling subsidies, or hard infrastructural fixes) risk maladaptation by increasing GHG emissions and eroding sustainable development, disproportionately burdening low‑income or otherwise vulnerable residents with energy costs, reducing incentives for low‑energy or behavioural cooling adaptations, and creating path dependencies and sunk costs that limit future flexible responses; these risks mirror documented maladaptation types and pathways including increased emissions, burdening the most vulnerable, reduced adaptation incentives, and path dependency shown in the literature (Evidence: Juhola et al., 2016; Chi et al., 2020; Neset et al., 2019).

# Inferred risk for: 구민 안전 확보 및 피해 최소화 – 체계적 풍수해 대응 관리 (ENG = Ensuring public safety and minimizing climate-related risks - Systematic management and response to storm and flood damage)
Systematic storm‑and‑flood response based on hard infrastructure, emergency relief and subsidies can create maladaptation by fostering a false sense of security that encourages development in protected areas and reduces individual incentives to adapt, thereby shifting or increasing vulnerability for less‑resourced residents (e.g., those unable to elevate foundations), generating path‑dependency and sunk costs for governments, and eroding long‑term sustainability through increased emissions and land‑use changes; these outcomes correspond to rebounding, shifting and eroding types of maladaptation and have been documented in case studies of intensified pump/dike/road works, emergency relief dependence, and post‑disaster subsidies and land‑conversion policies (Evidence: Chi et al., 2020; Juhola et al., 2016; Chi et al., 2021).

# Inferred risk for: 구민 안전 확보 및 피해 최소화 – 미세먼지 대응 강화 (ENG = Ensuring public safety and minimizing climate-related risks - Enhancing particulate matter mitigation)
Strengthening fine‑dust responses through predominantly technological or infrastructure fixes (e.g., large‑scale outdoor/indoor filtration systems, hard containment or industry relocation) risks creating maladaptation by generating path dependency and opportunity costs, increasing GHG emissions and energy use (thereby eroding sustainability), shifting or transferring risk onto other places or social groups (burdening low‑income or marginalized residents), and reducing incentives for emissions‑reducing policies—outcomes that may give a false sense of safety while increasing long‑term vulnerability and inequity (Evidence: Chi et al., 2020; Chi et al., 2021; Reckien et al., 2023).

# Inferred risk for: 재해로부터 안전한 산림환경 구축 – 산림종합방제 시스템 구축 (ENG = Building a safe forest environment protected from disasters - Establishing an integrated forest control system)
Implementing a centralized, techno‑centric integrated forest control system could be maladaptive if it is imposed top‑down without aligning to local mental models and development pathways, because it may create path‑dependence and high opportunity costs (locking resources into specific technologies or institutions), shift or rebound vulnerability onto other social groups or territories, erode broader sustainable development goals by privileging narrow technical objectives over local values, and reduce local adaptive capacity when socioeconomic drivers and participation are ignored (Evidence: Zango-Palau et al., 2024; Juhola et al., 2016; Findlater et al., 2021).

# Inferred risk for: 재해로부터 안전한 산림환경 구축 – 기후변화 적응을 위한 산림 확대 (ENG = Building a safe forest environment protected from disasters - Expanding forest cover for climate change adaptation)
Expanding forests as a climate‑adaptation measure can produce maladaptation by creating path dependence and high opportunity costs—locking policy and practice into particular species, technologies or commercial management models that foreclose broader governance options and non‑timber values—and by shifting harms onto other actors or common‑pool resources (e.g., water, soil) or producing technical failure when selected trees or interventions are mismatched to future conditions; stakeholders thus warn this can reinforce the status quo, reduce future adaptability, and externalize environmental and social trade‑offs (Evidence: Findlater et al., 2021; Juhola et al., 2016; Neset et al., 2019).

# Inferred risk for: 안정적 물이용 체계 확보 – 안전한 물 공급 및 깨끗한 수자원 관리 (ENG = Establishing a stable usage framework - Ensuring stable water supply and clean water resource management)
Measures to secure water supplies and manage water quality (e.g., environmental flows, engineered environmental works, water buybacks, and large infrastructure like desalination) can produce maladaptive trade‑offs: environmental flows and EWMs can spread invasive species and cause cold‑water pollution and uneven ecological benefits while water buybacks can disproportionately burden irrigation communities; energy‑intensive solutions such as desalination can increase GHG emissions and tie water security to energy‑sector vulnerabilities, creating path‑dependency and reducing incentives for conservation, thereby shifting vulnerability and eroding sustainable development (Evidence: Lukasiewicz et al., 2016; Juhola et al., 2016; Tubi and Williams, 2020).

# Inferred risk for: 기후변화 대응 역량 강화 – 저탄소 생활 실천 활성화 (ENG = Strengthening capacities for climate change response - Promoting the practice of low-carbon lifestyles)
Promoting low‑carbon lifestyle measures without contextualization or participatory deliberation can disproportionately burden low‑income or otherwise vulnerable groups (e.g., through higher upfront costs or opportunity costs), shift vulnerability onto marginalized actors, and erode adaptive capacity when top‑down, one‑size‑fits‑all policies conflict with local needs and mental models or divert resources from more effective adaptation, producing social inequity and potential maladaptive outcomes (Evidence: Juhola et al., 2016; Zango-Palau et al., 2024; Neset et al., 2019).
```

- Retrieval and re-ranking results are provided as JSON files for each objective–action pair.
- An example is shown below: 

```text
{
  "objective": "기후변화로부터 구민 건강 보호",
  "action": "취약계층 중심 건강관리 강화",
  "query": "Maladaptation from implementing '기후변화로부터 구민 건강 보호' via measure '취약계층 중심 건강관리 강화'.",
  "evidence": [
    {
      "evidence_id": "E2",
      "retrieval_rank": 5,
      "rerank_rank": 2,
      "rerank_score": 0.12791958451271057,
      "selected_for_llm": true,
      "citation": "Chi et al., 2021",
      "metadata": {
        "authors": "Chia-Fa Chi, Shiau-Yun Lu, Willow Hallgren, Daniel Ware, Rodger Tomlinson",
        "source": "C:\\path\\Included\\Chi et al. (2021).pdf",
        "doi": "10.3390/su13063450",
        "journal": "Sustainability",
        "year": "2021",
        "title": "Role of Spatial Analysis in Avoiding Climate Change Maladaptation: A Systematic Review"
      },
      "passage": null
   },

...
```

- Each record includes the objective, action, retrieval query, and retrieved evidence documents.
- For each document, it reports the evidence ID, original retrieval rank, re-ranking rank and score, selection status for LLM input, and bibliographic metadata.
- Article texts in the `passage` field are omitted from the publicly shared files to avoid redistributing copyrighted content.

## D. Operational workflow: End-user configuration 
### D.1. Summary

<p align="center">
  <img src="Figure_S1.png" width="800">
</p>

<p align="center">
  <b>Figure S1.</b> Model operation workflow for end users.
</p>

This section explains the operation of the proposed planning support model from an end-user perspective. To run the model, users must prepare the target planning document (PDF) and access the Upstage Document Parse and OpenAI (GPT) API key. The overall operational flow comprised three stages: (1) document input and preprocessing, (2) structured information extraction, and (3) evidence-based inference of maladaptation risks. This section outlines the general workflow of these procedures. Detailed code-level instructions are provided in the online Supplementary Material. 

First, the user submits the target plan as a PDF, which is converted into HTML via the Upstage Document Parse. As the parser cannot process documents exceeding 100 pages, any PDF exceeding this limit is split into 90-page segments (Figure S1(a)). Each segment is converted separately and the resulting HTML files are merged back into a document. To avoid the computational cost of scanning an entire document, the proposed model selectively reads only the relevant sections by searching for predefined keywords that appear after a user-specified starting page. Although this workflow is fully automated, users must still indicate the search starting point and keywords that guide the retrieval. 

Using the converted file, the model invokes the LLM with a predefined prompt (Figure S1(b)). The prompt instructs the model to classify specific expressions in the document as objectives or actions, consistent with the terminology used in the target planning document (e.g., implementation measure → action). It also prohibits the model from rewriting the content present in the document and inferring information that is not explicitly stated. Maladaptation risks are extracted at the level of each objective–action pair, and when no such risk is explicitly mentioned in the document, the model is required to record it as ‘(Missing).’ 

When potential maladaptation risks are not identified in the planning document, the user may activate an evidence-based inference module that leverages an external knowledge base (Figure 2(c)). At this stage, the user must perform an execution because the entire process is handled automatically, without controlling any internal procedures. The model generates a query for each objective–action pair, retrieves relevant literature from a Chroma vector database using bge-m3 embeddings (_k_=5), and applies a bge-reranker-v2-m3 reranker to select the highest-relevance texts (_k_=3) as the evidence.<sup>1</sup> Based on the selected evidence, the LLM infers a concise maladaptation and provides a citation at the end in the format. 

1) _k_ values are pragmatic settings to limit computational cost and are not theoretically fixed; they can be adjusted by corpus size and analytical objectives. 

### D.2. Section extraction parameters
- Users must know beforehand which page the search should start from and which keywords appear in the document. 
- The model begins scanning after the specified page and extracts the section once the keyword is detected, meaning that some prior knowledge is required.
- Used for target-section retrieval:

```python
min_page_threshold = 150
```

- Korean keywords: 

```python
start_kw = "부문별세부시행계획"
end_kw   = "계획의집행및관리"
```

- When the planning document changes, the keywords and the search starting point must be adjusted manually.

### D.3. Section extraction parameters
- The labels that correspond to {objective} and {action} must be specified for each document.
- In this study, the prompt explicitly specified which keywords in the document should be interpreted as {objective} and {action}:

```text
# Objective/Action classification rules:
## Only the sections in the document’s tables or main text that are explicitly marked as “추진전략” should be identified as {objective}.
## Only the sections in the document’s tables or main text that are explicitly marked as “실천과제” should be identified as {action}.
## Only structural labels (e.g., 추진전략, 실천과제) determine classification; Do not classify objectives and actions based on semantic meaning, wording style, and phrasing.
```
- Since local governments frequently employ different expressions for equivalent hierarchical concepts, {objective} and {action}, these labels were revised for each document to reflect its specific wording.
