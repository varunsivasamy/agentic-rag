# Agentic RAG

A single-file terminal CLI that lets you chat with a PDF using an **agentic** retrieval-augmented generation (RAG) pipeline. The app indexes `sample.pdf`, caches embeddings locally, and runs a LangChain agent powered by [Ollama](https://ollama.com/) with document search and calculator tools.

## Features

- **Agentic workflow** — The LLM decides when to retrieve document context vs. run calculations.
- **Local-first** — Uses Ollama for both chat (`qwen2.5:1.5b`) and embeddings (`nomic-embed-text`); no cloud API keys required.
- **Persistent embeddings** — ChromaDB vectors are saved in `./chroma_db` and reused across runs.
- **Smart re-indexing** — Automatically rebuilds the index when `sample.pdf` changes.
- **Conversation memory** — Multi-turn chat with in-session memory; use `clear` to reset.
- **MMR retrieval** — Maximal Marginal Relevance search for diverse, relevant chunks.
- **Safe calculator** — Math expressions are evaluated via AST parsing (no arbitrary code execution).

## How it works

```
sample.pdf  →  chunk & embed  →  ChromaDB (./chroma_db)
                                        ↓
User question  →  LangChain agent  →  document_retriever / calculator  →  answer
```

On startup, the app validates the PDF, checks Ollama connectivity, loads or builds the vector store, then starts an interactive chat loop.

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** installed and running locally
3. Pull the required models:

```bash
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

## Installation

```bash
git clone https://github.com/varunsivasamy/agentic-rag.git
cd agentic-rag

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install langchain langchain-chroma langchain-community langchain-ollama langchain-text-splitters chromadb pypdf
```

## Usage

Place your PDF in the project folder as `sample.pdf` (or replace the bundled file), then run:

```bash
python app.py
```

### Command-line options

| Flag | Description |
|------|-------------|
| `--reindex` | Force rebuild embeddings from `sample.pdf` |
| `--verbose` | Enable debug-level logging |

### Chat commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `clear` | Reset conversation memory |
| `exit` / `quit` | End the session |

### Example session

```
You: What is the main topic of the document?
Tools Used: document_retriever
--- Retrieved Context ---
[chunk 1, page 0] (sample.pdf): ...
-------------------------

Assistant: The document discusses ...
```

## Configuration

Key settings are defined at the top of `app.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_LLM_MODEL` | `qwen2.5:1.5b` | Chat / tool-calling model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MMR_K` | `5` | Number of chunks returned |
| `CHROMA_DIR` | `./chroma_db` | Vector store directory |

To use a different document, replace `sample.pdf` and restart with `--reindex`.

## Project structure

```
agentic-rag/
├── app.py          # Full application (CLI, RAG pipeline, agent)
├── sample.pdf      # Default document to index and query
├── chroma_db/      # Cached embeddings (created at runtime, gitignored)
└── README.md
```

## Troubleshooting

**Cannot connect to Ollama**

- Ensure the Ollama service is running.
- Confirm both models are pulled (see Prerequisites).

**ChromaDB collection is empty**

- Run `python app.py --reindex` to rebuild embeddings.

**PDF contains no extractable text**

- The PDF may be image-only; use a text-based PDF or OCR the document first.

## License

This project is provided as-is for learning and experimentation.
