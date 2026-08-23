"""
LangChain ReAct Agent — decides at each turn whether to:
  1. Use document_retriever (RAG) for company policy questions
  2. Use calculator for math
  3. Answer directly from LLM for general conversation
"""

import ast
import json
import logging
import operator
import os
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
# CONFIG
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


# =============================================================================
# SAFE CALCULATOR TOOL
# =============================================================================

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"Unsupported: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Perform safe math calculations. Input must be a valid math expression like '10 * 3 + 5'."""
    try:
        result = _eval_node(ast.parse(expression.strip(), mode="eval").body)
        return str(int(result)) if result == int(result) else str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# PDF INGESTION
# =============================================================================

def _get_pdf_files() -> list[str]:
    if not os.path.isdir(PDF_FOLDER):
        raise RuntimeError(f"PDF folder not found: {PDF_FOLDER}")
    pdfs = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {PDF_FOLDER}")
    return pdfs


def _fingerprint(pdf_files: list[str]) -> dict:
    return {
        "files": {
            os.path.abspath(p): {"mtime": os.stat(p).st_mtime, "size": os.stat(p).st_size}
            for p in sorted(pdf_files)
        },
        "embed_model": OLLAMA_EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def _needs_reindex(pdf_files: list[str]) -> bool:
    if not os.path.isdir(CHROMA_DIR) or not os.path.isfile(INDEX_META_FILE):
        return True
    try:
        with open(INDEX_META_FILE, encoding="utf-8") as f:
            return json.load(f) != _fingerprint(pdf_files)
    except Exception:
        return True


def load_and_index_pdfs(force: bool = False) -> Chroma:
    """Load all PDFs from ./pdf, chunk, embed, and store in ChromaDB."""
    pdf_files = _get_pdf_files()
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    if not force and not _needs_reindex(pdf_files):
        logger.info("Using cached embeddings from %s", CHROMA_DIR)
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

    logger.info("Indexing %d PDF(s)...", len(pdf_files))
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []

    for pdf_path in pdf_files:
        docs = PyPDFLoader(pdf_path).load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        logger.info("  %s → %d chunks", os.path.basename(pdf_path), len(chunks))

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(_fingerprint(pdf_files), f, indent=2)

    logger.info("Vector store ready. Total chunks: %d", len(all_chunks))
    return vectorstore


SYSTEM_PROMPT = """You are a smart company assistant. You have been given access to all the company's internal documents — these cover HR policies, code of conduct, IT & data security, health & safety, travel, communications, branding, and more.

Behave as if the user has handed you all those PDFs directly in the chat. Use them to answer ANY company-related question — not just HR. If it's about the company, search for it.

You have access to:
1. document_retriever — searches ALL indexed company documents
2. calculator — performs math calculations

Decision rules:
* If the question is about ANYTHING related to the company (policies, rules, procedures, benefits, roles, IT, travel, conduct, branding, safety, etc.) → call document_retriever FIRST, then answer.
* If the question involves numbers or calculations → use calculator.
* If it is casual conversation, a greeting, or completely unrelated to the company → answer naturally without using any tool.
* Never make up information. If the documents don't contain the answer, say so clearly.
* When you use document_retriever, always tell the user which document the answer came from."""


# =============================================================================
# AGENT FACTORY
# =============================================================================

def build_agent(vectorstore: Chroma):
    """Build a LangChain agent with document_retriever and calculator tools."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
    )

    retriever_tool = create_retriever_tool(
        retriever,
        name="document_retriever",
        description=(
            "Search ALL company documents including HR policies, employee handbook, "
            "code of conduct, IT and data security policy, health and safety, travel policy, "
            "communications, branding guidelines, and any other company-specific information. "
            "Use this tool for ANY question that might be answered by the company's documents."
        ),
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

    logger.info("Agent ready. Tools: %s", [t.name for t in tools])
    return agent


# =============================================================================
# QUERY
# =============================================================================

def query_agent(agent, thread_id: str, question: str, history: list[dict]) -> dict[str, Any]:
    """Run a question through the agent with conversation history via thread_id."""
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
                    entry = f"{src} (p.{page})"
                    if entry not in sources:
                        sources.append(entry)

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
