# Document RAG with Streamlit and Groq

Upload PDF/DOCX documents and ask questions grounded in retrieved excerpts.
Uses the open-source sentence-transformers/all-MiniLM-L6-v2 embedding model on
the Streamlit server and the open-weight openai/gpt-oss-20b model hosted by Groq.
The OpenAI Python package is just the compatible client; you need a Groq key,
not an OpenAI key. No GPU is required for this app server.

## 1. Understand the flow

Indexing: upload -> extract text -> tokenize with the embedding tokenizer ->
split into overlapping token windows -> embed chunks -> keep vectors and source
metadata in the user's Streamlit session.

Answering: question -> same embedding model -> cosine similarity against stored
vectors -> top matching chunks -> Groq Responses API -> answer and source excerpts.

Tokenization converts text into token IDs. Embedding converts tokenized text into
numeric vectors representing meaning. Token-aware splitting uses tokenization
to decide chunk boundaries; the embedding library tokenizes each chunk again
internally. Groq applies its own generation tokenizer. The two token counts need
not match. This is retrieval, not model training or fine-tuning.

Default chunks contain at most 220 embedding tokens with 40-token overlap.
This fits MiniLM's 256-token input limit, including special tokens. Original
character spans preserve case, punctuation, amounts and dates. Chunks never cross
PDF page or Word body-block boundaries, so many chunks will be smaller.
Normalized embeddings have 384 dimensions; NumPy implements an in-memory vector
index using cosine similarity. No separate vector database is needed for this demo.

## 2. Get a Groq API key

Create an account at https://console.groq.com/keys and create an API key. Keep it
private. The default model is openai/gpt-oss-20b; you can change the model in the
sidebar to another model supported by your Groq account.

## 3. Create the GitHub repository

On GitHub, choose New repository, name it `rag-document-app`, and create it.
Use Add file -> Upload files to upload `app.py`, `requirements.txt`, `.gitignore`,
and this README. Commit the files to your main branch.

Use the exact filename `requirements.txt` (plural) for Streamlit's automatic
dependency installation. `requirement.txt` is an identical optional copy supplied
because it was requested; it is not necessary in the repository.

Do not upload API keys, `.streamlit/secrets.toml`, your virtual environment,
or confidential source documents. The app accepts documents through its uploader.

## 4. Run locally (optional but recommended)

Install Python 3.11 and download/clone your repository. Open a terminal in its folder.

Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Enter your Groq key in the app's password field. Alternatively, create the local
file `.streamlit/secrets.toml` containing:

```toml
GROQ_API_KEY = "paste_your_actual_groq_key_here"
```

An environment variable named GROQ_API_KEY is also supported and takes precedence.
The initial build downloads the embedding model from Hugging Face; internet
access is required. Later builds reuse cached weights. Sentence Transformers
installs PyTorch as a dependency, so initial installation can be substantial.
The requirements file uses bounded compatible version ranges, not a tested lockfile.

## 5. Deploy on Streamlit Community Cloud

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Create an app and select your repository and `main` branch.
3. Set the main file path to `app.py`.
4. In Advanced settings, select Python 3.11 if offered.
5. In Secrets, paste the TOML GROQ_API_KEY entry shown above, with your real key.
6. Deploy. Wait for dependencies to install and the app to start.
7. Open the app, upload a document, and click Build document index.

Secrets are configured in Streamlit, never in the GitHub source. If you plan to
share the app, choose the intended access settings: visitors using your configured
server key consume your Groq quota. You can omit the server key and have visitors
enter their own key. This starter does not implement authentication or a persistent
database. Hosting may require a larger memory allowance for PyTorch and parsing.

## 6. Use and validate the app

Start with a small text-based PDF or DOCX with a known fact such as a 90-day
completion period. Ask "What is the completion period?" and check the cited
page/block in View retrieved evidence. Then ask about a fact absent from the
document and check that the model says it cannot find it. Test a Word table,
multiple files, and replacing/removing an uploaded file. File changes invalidate
the previous index and displayed answers.

Example engineering questions:
- What are the payment milestones?
- What completion period is specified?
- Who is responsible for supplying materials?
- What inspection requirements are stated?

Ask each question in full. Chat history is displayed (last 10 exchanges), but
previous turns are not sent to the model and do not influence retrieval.

## Limits and troubleshooting

- PDF text and DOCX body paragraphs/tables are supported. Legacy DOC requires
  conversion to DOCX. Scanned pages require OCR before upload. Headers, footers,
  text boxes, images and embedded objects are not extracted from Word documents.
- PDF columns and complex tables can extract out of order. Verify source excerpts
  before relying on amounts or clauses. DOCX citations use body blocks, not pages.
- Up to five files, 20 MB combined, two million extracted characters and 12,000
  chunks. These limits are starter guardrails, not a hardened document sandbox.
- The embedding model is intended primarily for English. Use a multilingual
  embedding model and adjust its chunk limit for multilingual document retrieval.
- The index is session-only and is lost on session termination/server restart.
  Documents are not put in Streamlit's shared data cache. Only model weights are
  cached across sessions. The question and retrieved excerpts leave the app server
  for Groq; embedding alone does not send documents to Groq.
- Top-k retrieval may miss relevant sections and is not a full-document audit or
  exhaustive summary. Similarity scores are not confidence probabilities.
  Grounding instructions reduce hallucinations but do not guarantee factual
  correctness or prompt-injection immunity. Inspect the evidence.
- Authentication error: verify the Groq key. Rate limit: wait or reduce retrieved
  chunks. Model error: check availability/access in the Groq console. Indexing
  error: check file validity, memory, and access to Hugging Face for model weights.

## Validation performed on the delivered source

Passed Python syntax validation; real PDF and DOCX extraction with a Word table;
chunk coverage and overlap using a deterministic test tokenizer; and similarity
ranking using fixed test embeddings. Full Streamlit execution, real embedding
downloads, dependency installation and live Groq requests were not tested in the
authoring environment. Run the small-document checks above after deployment.

## Official references

- Groq Responses API: https://console.groq.com/docs/responses-api
- Groq model: https://console.groq.com/docs/model/openai/gpt-oss-20b
- Embedding model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Streamlit dependencies: https://docs.streamlit.io/deploy/concepts/dependencies
- Streamlit secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Python selection: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python
