"""
LangChain Agent with Groq LLM.

Uses LangChain's bind_tools + tool-calling chain — no LangGraph.
The LLM decides at each turn:
  1. Call document_retriever  -> answer from company PDFs
  2. Call calculator           -> math
  3. Answer directly           -> general conversation
"""

import ast
import json
import logging
import operator
import os
import shutil
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, create_retriever_tool
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("rag_engine")

load_dotenv()  # loads GROQ_API_KEY from .env

# =============================================================================
# CONFIG
# =============================================================================

GROQ_API_KEY   = os.environ["GROQ_API_KEY"]
GROQ_LLM_MODEL = "llama3-70b-8192"

EMBED_MODEL  = "nomic-embed-text"     # still local via Ollama
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "company_docs"
INDEX_META   = os.path.join(CHROMA_DIR, "index_meta.json")
PDF_FOLDER   = "./pdf"
MMR_K        = 5
MMR_FETCH_K  = 20

SYSTEM_PROMPT = """You are a knowledgeable company assistant. You have been given access to all the company's internal documents — HR policies, code of conduct, IT & data security, health & safety, travel, communications, branding, and more.

Treat the documents as if the user handed them to you directly. Use them to answer ANY company-related question.

You have two tools:
1. document_retriever — searches ALL company documents. Use for any company or policy question.
2. calculator — math calculations.

Rules:
- Company question? → call document_retriever first, then answer using the retrieved content.
- Math? → use calculator.
- Casual chat / greeting? → answer directly, no tool needed.
- Never make up company information. If the docs don't contain it, say so.
- Always tell the user which document your answer came from."""


# =============================================================================
# SAFE CALCULATOR
# =============================================================================

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError(f"Unsupported: {ast.dump(node)}")

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression like '10 * 3 + 5'."""
    try:
        r = _eval(ast.parse(expression.strip(), mode="eval").body)
        return str(int(r)) if r == int(r) else str(r)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# PDF INGESTION
# =============================================================================

def _pdf_files() -> list[str]:
    if not os.path.isdir(PDF_FOLDER):
        raise RuntimeError(f"PDF folder not found: {PDF_FOLDER}")
    files = [os.path.join(PDF_FOLDER, f)
             for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not files:
        raise RuntimeError(f"No PDFs in {PDF_FOLDER}")
    return files

def _fingerprint(files: list[str]) -> dict:
    return {
        "files": {os.path.abspath(p): {"mtime": os.stat(p).st_mtime,
                                        "size":  os.stat(p).st_size}
                  for p in sorted(files)},
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

def _needs_reindex(files: list[str]) -> bool:
    if not os.path.isdir(CHROMA_DIR) or not os.path.isfile(INDEX_META):
        return True
    try:
        return json.load(open(INDEX_META, encoding="utf-8")) != _fingerprint(files)
    except Exception:
        return True

def load_and_index_pdfs(force: bool = False) -> Chroma:
    files = _pdf_files()
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if not force and not _needs_reindex(files):
        logger.info("Using cached embeddings.")
        return Chroma(collection_name=COLLECTION,
                      embedding_function=embeddings,
                      persist_directory=CHROMA_DIR)

    logger.info("Indexing %d PDFs...", len(files))
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for p in files:
        docs = PyPDFLoader(p).load()
        c = splitter.split_documents(docs)
        chunks.extend(c)
        logger.info("  %s → %d chunks", os.path.basename(p), len(c))

    vs = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                collection_name=COLLECTION,
                                persist_directory=CHROMA_DIR)
    json.dump(_fingerprint(files), open(INDEX_META, "w", encoding="utf-8"), indent=2)
    logger.info("Indexed %d total chunks.", len(chunks))
    return vs


# =============================================================================
# AGENT — pure LangChain tool-calling chain
# =============================================================================

class Agent:
    """
    Stateless LangChain tool-calling agent.
    Conversation history is passed in on each call.
    """

    def __init__(self, vectorstore: Chroma):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
        )
        self.retriever_tool = create_retriever_tool(
            retriever,
            name="document_retriever",
            description=(
                "Search ALL company documents: HR policies, employee handbook, "
                "code of conduct, IT & data security, health & safety, travel, "
                "communications, branding. Use for ANY company-related question."
            ),
        )
        self.tools      = [self.retriever_tool, calculator]
        self.tools_map  = {t.name: t for t in self.tools}
        self.llm        = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_LLM_MODEL,
            temperature=0.2,
        ).bind_tools(self.tools)

    def run(self, question: str, history: list[dict]) -> dict[str, Any]:
        """Run one turn. history = list of {role, content} dicts."""

        # Build message list
        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history[-10:]:           # last 5 turns
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))

        tools_used: list[str] = []
        sources:    list[str] = []

        # Agentic loop — up to 5 iterations
        for _ in range(5):
            response: AIMessage = self.llm.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                # No tool call → final answer
                break

            # Execute each tool call
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                tool_fn = self.tools_map.get(tool_name)
                if tool_fn is None:
                    result = f"Unknown tool: {tool_name}"
                else:
                    result = tool_fn.invoke(tool_args)

                # Extract sources from retriever results
                if tool_name == "document_retriever" and isinstance(result, str):
                    # result is a formatted string of doc chunks
                    for line in result.split("\n\n"):
                        if "source" in line.lower() or ".pdf" in line.lower():
                            for part in line.split():
                                if ".pdf" in part.lower():
                                    src = os.path.basename(part.strip("()[],'\""))
                                    if src and src not in sources:
                                        sources.append(src)

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        # Extract final text reply
        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                reply = msg.content if isinstance(msg.content, str) else \
                    " ".join(b.get("text", "") for b in msg.content
                             if isinstance(b, dict) and b.get("type") == "text")
                if reply:
                    break

        return {
            "reply":      reply or "I could not generate a response.",
            "tools_used": tools_used,
            "sources":    sources,
        }


def build_agent(vectorstore: Chroma) -> "Agent":
    agent = Agent(vectorstore)
    logger.info("Groq agent ready. Tools: %s", [t.name for t in agent.tools])
    return agent


def query_agent(agent: "Agent", thread_id: str,
                question: str, history: list[dict]) -> dict[str, Any]:
    return agent.run(question, history)
