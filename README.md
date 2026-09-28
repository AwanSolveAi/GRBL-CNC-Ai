# GRBL CNC AI — Technical Support Assistant

GRBL CNC AI is a local, documentation-grounded assistant for GRBL CNC support.
It provides fast answers for exact identifiers and uses local retrieval and
generation for broader troubleshooting questions—without a paid API.

## Key capabilities

- Fast deterministic answers for exact settings, G-codes, M-codes, and system commands
- Hybrid BM25 + FAISS retrieval
- Cross-encoder reranking for troubleshooting and ambiguous questions
- Local answer generation through Ollama
- Deterministic, grounded source citations
- CNC safety guardrails
- Refusal when the bundled documentation does not support an answer
- Professional local Streamlit browser demo

## Validated performance

- Exact-query backend latency: approximately **14 ms** after initialization
- Fast regression suite: **23/23 passing**
- Core retrieval evaluation: **44/44**
- Answer-quality evaluation: **28/28**
- Adversarial answer evaluation: **24/24**
- Citation faithfulness: **52/52**
- One adversarial Top-1 typo miss is documented; the correct evidence ranked second

## Requirements

- Windows
- Python 3.11
- Ollama running locally
- Ollama model `llama3.2:3b`
- Bundled processed corpus and local FAISS/index assets

Expected data assets:

- `data/processed/grbl_corpus.json`
- `data/vector_store/grbl_faiss.index`
- `data/vector_store/metadata.json`
- `data/vector_store/index_manifest.json`

## Setup

From PowerShell in the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
```

Start Ollama if it is not already running:

```powershell
ollama serve
```

## Launch

Double-click `run_demo.bat`, or run:

```powershell
.\run_demo.bat
```

Direct Streamlit command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in a browser.

CLI launch:

```powershell
.\.venv\Scripts\python.exe src\grbl_ai.py
```

## Configuration

Configuration is read from environment variables at process startup. No API
keys are required.

| Variable | Default | Purpose |
|---|---:|---|
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2:3b` | Local generation model |
| `GRBL_OLLAMA_NUM_PREDICT` | `200` | Maximum generated tokens |
| `GRBL_FORCE_RERANK_ALL` | `false` | Force reranking for every query |
| `GRBL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model matching the index |
| `GRBL_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `GRBL_VECTOR_TOP_K` | `20` | Vector candidates |
| `GRBL_BM25_TOP_K` | `20` | BM25 candidates |
| `GRBL_HYBRID_TOP_K` | `20` | Fused candidates |
| `GRBL_RERANK_TOP_K` | `10` | Reranked candidates |
| `GRBL_ANSWER_CONTEXT_K` | `5` | Answer context chunks |

`.env.example` is a reference only; the project does not require or load
`python-dotenv`. Set overrides in the shell or operating-system environment.

## Startup troubleshooting

- Confirm Ollama is running before asking troubleshooting/general questions.
- Confirm `llama3.2:3b` is installed with `ollama list`.
- The first embedding or reranker initialization may take several seconds.
- If optional vector components cannot initialize, the assistant reports degraded mode and uses BM25 where possible.

## Known limitations

- Official numbered GRBL alarm definitions are not yet included in the corpus.
- Troubleshooting generation is slower than deterministic exact lookup.
- The current corpus contains 102 chunks.
- Local CPU performance depends on the client hardware.

See [docs/demo_guide.md](docs/demo_guide.md) for the recommended presentation flow.
