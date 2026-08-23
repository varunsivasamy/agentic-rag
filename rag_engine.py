"""
RAG Engine — loads all PDFs from the ./pdf folder, builds ChromaDB embeddings,
and exposes a query method used by the FastAPI backend.
"""

import ast
import logging
import operator
import os
import json
import shutil
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import create_retriever_tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger("rag_engine")

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "company_docs"
INDEX_META_FILE = os.path.join(CHROMA_DIR, "index_meta.json")
PDF_FOLDER = "./pdf"

OLLAMA_LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
LLM_TEMPERATURE = 0.2
MMR_K = 5
MMR_FETCH_K = 20

SYSTEM_PROMPT = """You are an intelligent HR and company policy assistant.

All company PDF documents are already indexed and available to you.
Do NOT ask the user to upload any files.

You have access to:
1. document_retriever — searches indexed company PDFs for relevant information
2. calculator — performs math calculations

Rules:
* For ANY question about policies, benefits, leave, roles, or company rules, call document_retriever FIRST.
* Use calculator for numerical computations.
* Never hallucinate. Only answer using retrieved content.
* If information is not found, clearly say so.
* Always mention which document the answer came from.
* Keep responses concise, accurate, and professional."""


# =============================================================================
# SAFE CALCULATOR
# =============================================================================

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval_node(node.left),
            _safe_eval_node(node.right),
        )
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations accurately."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_node(tree.body)
        return str(int(result)) if result == int(result) else str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        return f"Error: Could not evaluate. {exc}"


# =============================================================================
# PDF INGESTION
# =============================================================================

def _get_pdf_files() -> list[str]:
    """Return all PDF paths from the pdf folder."""
    if not os.path.isdir(PDF_FOLDER):
        raise RuntimeError(f"PDF folder not found: {PDF_FOLDER}")
    pdfs = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {PDF_FOLDER}")
    return pdfs


def _build_fingerprint(pdf_files: list[str]) -> dict:
    """Fingerprint all PDFs for cache invalidation."""
    files_meta = {}
    for p in sorted(pdf_files):
        stat = os.stat(p)
        files_meta[os.path.abspath(p)] = {
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
    return {
        "files": files_meta,
        "embed_model": OLLAMA_EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def _needs_reindex(pdf_files: list[str]) -> bool:
    if not os.path.isdir(CHROMA_DIR):
        return True
    if not os.path.isfile(INDEX_META_FILE):
        return True
    try:
        with open(INDEX_META_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        return saved != _build_fingerprint(pdf_files)
    except Exception:
        return True


def _write_index_meta(pdf_files: list[str]) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(_build_fingerprint(pdf_files), f, indent=2)


def load_and_index_pdfs(force: bool = False) -> Chroma:
    """Load all PDFs from ./pdf, chunk, embed, and store in ChromaDB."""
    pdf_files = _get_pdf_files()
    logger.info("Found %d PDF(s) in %s", len(pdf_files), PDF_FOLDER)

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    if not force and not _needs_reindex(pdf_files):
        logger.info("Using cached embeddings from %s", CHROMA_DIR)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

    logger.info("Building new vector store...")
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path)
        docs = PyPDFLoader(pdf_path).load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        logger.info("  → %d chunks", len(chunks))

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    _write_index_meta(pdf_files)
    logger.info("Vector store built with %d total chunks.", len(all_chunks))
    return vectorstore


# =============================================================================
# AGENT
# =============================================================================

def build_agent(vectorstore: Chroma):
    """Build LangChain agent with document retriever and calculator tools."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="document_retriever",
        description="Search company PDF documents and retrieve relevant policy information.",
        response_format="content_and_artifact",
    )
    tools = [retriever_tool, calculator]

    llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=LLM_TEMPERATURE)
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    logger.info("Agent ready with tools: %s", [t.name for t in tools])
    return agent


# =============================================================================
# QUERY
# =============================================================================

def query_agent(agent, thread_id: str, question: str) -> dict[str, Any]:
    """Run a question through the agent and return structured response."""
    config = {"configurable": {"thread_id": thread_id}}

    prior_state = agent.get_state(config)
    prior_count = len(prior_state.values.get("messages", []))

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    messages = result.get("messages", [])[prior_count:]
    tools_used, sources, reply = [], [], ""

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "unknown")
                if name not in tools_used:
                    tools_used.append(name)

        if isinstance(msg, ToolMessage) and msg.name == "document_retriever":
            artifact = getattr(msg, "artifact", None)
            if artifact:
                for doc in artifact:
                    src = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    if f"{src} (p.{page})" not in sources:
                        sources.append(f"{src} (p.{page})")

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content if isinstance(msg.content, str) else " ".join(
                b.get("text", "") for b in msg.content if isinstance(b, dict)
            )
            if reply:
                break

    return {
        "reply": reply or "I could not generate a response.",
        "tools_used": tools_used,
        "sources": sources,
    }
