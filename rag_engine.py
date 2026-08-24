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
GROQ_LLM_MODEL = "openai/gpt-oss-120b"  # reliable tool-calling on this account

EMBED_MODEL  = "nomic-embed-text"     # still local via Ollama
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "company_docs"
INDEX_META   = os.path.join(CHROMA_DIR, "index_meta.json")
PDF_FOLDER   = "./pdf"
MMR_K        = 5
MMR_FETCH_K  = 20

SYSTEM_PROMPT = """You are a company assistant with access to all internal company documents.

When answering questions about company policies or information:
- Be concise and direct — 3 to 6 bullet points maximum
- Only include what is explicitly stated in the retrieved documents
- Do not add generic advice, filler, or content not in the documents
- End with one line stating the source document name
- If the document does not contain the answer, say so in one sentence

For casual conversation, reply naturally in one or two sentences. No tools needed."""


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
            temperature=0,
        ).bind_tools(self.tools, tool_choice="auto")

    def run(self, question: str, history: list[dict]) -> dict[str, Any]:
        """Run one turn. history = list of {role, content} dicts."""

        # Build message list
        messages: list = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in history[-10:]:
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
                break

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                tool_fn = self.tools_map.get(tool_name)
                result  = tool_fn.invoke(tool_args) if tool_fn else f"Unknown tool: {tool_name}"

                if tool_name == "document_retriever" and isinstance(result, str):
                    for line in result.split("\n\n"):
                        for part in line.split():
                            if ".pdf" in part.lower():
                                src = os.path.basename(part.strip("()[],'\""))
                                if src and src not in sources:
                                    sources.append(src)

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))

        # Extract reply — skip AIMessages that only contain tool calls
        def _extract_content(msg: AIMessage) -> str:
            c = msg.content
            if isinstance(c, list):
                c = " ".join(b.get("text", "") for b in c
                             if isinstance(b, dict) and b.get("type") == "text")
            c = (c or "").strip()
            # Strip <think>...</think> reasoning blocks (some models include these)
            import re
            c = re.sub(r"<think>.*?</think>", "", c, flags=re.DOTALL).strip()
            return c

        reply = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                reply = _extract_content(msg)
                if reply:
                    break

        # Fallback: qwen3 sometimes returns empty string after tool results.
        # Ask the plain LLM (no tools bound) to summarize the retrieved content.
        if not reply and tools_used:
            logger.info("Empty reply after tool use — invoking fallback summary call.")
            # Get the base LLM without tools bound
            base_llm = self.llm.bound if hasattr(self.llm, "bound") else self.llm
            messages.append(HumanMessage(
                content="Using ONLY the information retrieved above, provide a clear and complete answer to my original question."
            ))
            fallback = base_llm.invoke(messages)
            reply = _extract_content(fallback)

        return {
            "reply":      reply or "Sorry, I could not find relevant information.",
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
