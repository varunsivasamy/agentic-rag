# Employee Handbook Assistant

An agentic RAG application that extracts and answers questions from private company data — the **Company Employee Handbook** — using LangChain, ChromaDB, and Hugging Face models.

## Features

- **Private data extraction** — Queries internal company handbook, not a generic chatbot.
- **Agentic workflow** — The LLM decides when to retrieve document context vs. run calculations.
- **Grounded answers** — Responses are strictly based on handbook content; no hallucination.
- **Conversation memory** — Multi-turn chat with in-session memory.
- **MMR retrieval** — Maximal Marginal Relevance search for diverse, relevant chunks.

## How it works

```
Company_Employee_Handbook.pdf  →  chunk & embed  →  ChromaDB
                                                          ↓
User question  →  LangChain agent  →  document_retriever / calculator  →  answer
```

The handbook is indexed on startup. Users can immediately ask about policies, benefits, leave rules, and more.

## Prerequisites

- Python 3.10+
- `Company_Employee_Handbook.pdf` placed in the project folder

## Installation

```bash
git clone https://github.com/varunsivasamy/agentic-rag.git
cd agentic-rag

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The handbook loads automatically on startup. Ask questions like:

- *What is the leave policy?*
- *What are the employee benefits?*
- *What is the code of conduct?*

## Project structure

```
agentic-rag/
├── app.py                        # Main application
├── Company_Employee_Handbook.pdf # Private company data source
├── chroma_db/                    # Cached embeddings (gitignored)
└── README.md
```

## License

This project is provided as-is for learning and experimentation.
