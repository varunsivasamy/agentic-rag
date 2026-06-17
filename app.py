"""
Agentic RAG Application — Single-file terminal CLI.

Loads sample.pdf from the project folder, caches embeddings in ./chroma_db,
and runs a LangChain agent (qwen2.5:1.5b) with document_retriever
and calculator tools plus conversation memory.
"""

# =============================================================================
# SECTION 1: IMPORTS AND LOGGING
# =============================================================================

import argparse
import ast
import json
import logging
import operator
import os
import shutil
import sys
import uuid
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

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agentic_rag")


# =============================================================================
# SECTION 2: CONFIGURATION CONSTANTS
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = "./chroma_db"
INDEX_META_FILE = os.path.join(CHROMA_DIR, "index_meta.json")
COLLECTION_NAME = "pdf_documents"

# Fixed PDF in the project folder (change sample.pdf to update the document)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "sample.pdf")

# Small-model stack (~1.3 GB total) — well suited for local RAG:
#   qwen2.5:1.5b      ~1.0 GB  — tool calling, good reasoning for its size
#   nomic-embed-text  ~274 MB  — strong retrieval quality, tiny footprint
# Alternatives: llama3.2:1b (~1.3 GB LLM), all-minilm (~46 MB embed, lower quality)
OLLAMA_LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
LLM_TEMPERATURE = 0.2
MMR_K = 5
MMR_FETCH_K = 20


# =============================================================================
# SECTION 3: SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an intelligent document assistant.

The user's PDF is ALREADY uploaded and indexed in the vector database.
You do NOT need to ask the user to upload any file.

You have access to two tools:

1. document_retriever — searches the indexed PDF for relevant passages
2. calculator — performs math calculations

Rules:

* Always determine the user's intent first.
* For ANY question about the document, PDF, or its contents, you MUST call document_retriever FIRST before answering.
* Use calculator for mathematical operations.
* Never hallucinate. Only answer document questions using retrieved context.
* If information is unavailable in the retrieved context, clearly say so.
* Never tell the user to upload a file — the document is already available via document_retriever.
* Keep responses concise and accurate."""


# =============================================================================
# SECTION 4: ERROR HANDLING HELPERS
# =============================================================================

class AppError(Exception):
    """User-facing application error with a clear message."""


def validate_pdf_path(pdf_path: str) -> str:
    """Validate that the PDF path exists and is a non-empty file."""
    resolved = os.path.abspath(pdf_path)

    if not os.path.isfile(resolved):
        raise AppError(f"PDF file not found: {resolved}")

    if not resolved.lower().endswith(".pdf"):
        raise AppError(f"Expected a .pdf file, got: {resolved}")

    if os.path.getsize(resolved) == 0:
        raise AppError(f"PDF file is empty: {resolved}")

    return resolved


def _pdf_fingerprint(pdf_path: str) -> dict:
    """Build a fingerprint of the PDF file for cache invalidation."""
    stat = os.stat(pdf_path)
    return {
        "pdf_path": os.path.abspath(pdf_path),
        "pdf_mtime": stat.st_mtime,
        "pdf_size": stat.st_size,
        "embed_model": OLLAMA_EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


def _read_index_meta() -> dict | None:
    """Read saved index metadata if it exists."""
    if not os.path.isfile(INDEX_META_FILE):
        return None
    try:
        with open(INDEX_META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read index metadata: %s", exc)
        return None


def _write_index_meta(pdf_path: str) -> None:
    """Persist metadata after a successful index build."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(_pdf_fingerprint(pdf_path), f, indent=2)
    logger.info("Saved index metadata to %s", INDEX_META_FILE)


def needs_reindex(pdf_path: str) -> bool:
    """Return True if embeddings must be rebuilt from the PDF."""
    if not os.path.isdir(CHROMA_DIR):
        return True

    saved = _read_index_meta()
    if saved is None:
        return True

    return saved != _pdf_fingerprint(pdf_path)


def get_embeddings() -> OllamaEmbeddings:
    """Return the configured Ollama embedding model."""
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        if vectorstore._collection.count() == 0:
            raise AppError("ChromaDB collection is empty.")
        logger.info("Loaded existing vector store from %s", CHROMA_DIR)
        return vectorstore
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            f"Failed to load vector store from {CHROMA_DIR}: {exc}\n"
            "Run with --reindex to rebuild embeddings."
        ) from exc


def check_ollama_connection() -> None:
    """Verify Ollama is reachable by running a lightweight embedding call."""
    logger.info("Checking Ollama connection...")
    try:
        get_embeddings().embed_query("connection test")
        logger.info("Ollama connection successful.")
    except Exception as exc:
        raise AppError(
            "Cannot connect to Ollama. Ensure Ollama is running and models are pulled:\n"
            "  ollama pull qwen2.5:1.5b\n"
            "  ollama pull nomic-embed-text\n"
            f"Details: {exc}"
        ) from exc


def safe_process_pdf(pdf_path: str) -> list:
    """Load and split a PDF into text chunks, with validation."""
    logger.info("Loading PDF: %s", pdf_path)
    try:
        documents = PyPDFLoader(pdf_path).load()
    except Exception as exc:
        raise AppError(f"Failed to read PDF: {exc}") from exc

    if not documents:
        raise AppError("The PDF contains no pages.")

    non_empty_pages = [
        doc for doc in documents if doc.page_content and doc.page_content.strip()
    ]
    if not non_empty_pages:
        raise AppError("The PDF contains no extractable text.")

    logger.info("Loaded %d page(s). Splitting into chunks...", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    try:
        chunks = splitter.split_documents(documents)
    except Exception as exc:
        raise AppError(f"Failed to split PDF text: {exc}") from exc

    if not chunks:
        raise AppError("Text splitting produced zero chunks.")

    logger.info("Created %d chunk(s).", len(chunks))
    return chunks


def safe_build_vectorstore(chunks: list, pdf_path: str) -> Chroma:
    """Build a persisted ChromaDB vector store from document chunks."""
    logger.info("Building new vector store at %s", CHROMA_DIR)
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
        _write_index_meta(pdf_path)
        logger.info("Vector store created and saved successfully.")
        return vectorstore
    except Exception as exc:
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        raise AppError(
            f"ChromaDB error while building vector store: {exc}\n"
            "Try running with --reindex after fixing the issue."
        ) from exc


# =============================================================================
# SECTION 5: PDF PROCESSING PIPELINE (orchestration helper)
# =============================================================================

def index_pdf(pdf_path: str) -> Chroma:
    """Full ingestion pipeline: load PDF, split, embed, and store in ChromaDB."""
    chunks = safe_process_pdf(pdf_path)
    return safe_build_vectorstore(chunks, pdf_path)


def load_or_index_vectorstore(pdf_path: str, force_reindex: bool = False) -> Chroma:
    """
    Load cached embeddings from chroma_db, or index the PDF only when needed.

    Re-indexes automatically when sample.pdf changes or --reindex is passed.
    """
    if force_reindex or needs_reindex(pdf_path):
        if force_reindex:
            logger.info("Forced re-index requested.")
        else:
            logger.info("PDF changed or no cache found — indexing required.")
        return index_pdf(pdf_path)

    print(f"Using cached embeddings from {CHROMA_DIR} (sample.pdf unchanged).")
    return load_vectorstore()


# =============================================================================
# SECTION 6: VECTOR STORE AND MMR RETRIEVER
# =============================================================================

def create_mmr_retriever(vectorstore: Chroma):
    """Create an MMR retriever with the configured search parameters."""
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "fetch_k": MMR_FETCH_K},
    )


# =============================================================================
# SECTION 7: TOOL DEFINITIONS
# =============================================================================

# Safe math operators for the calculator tool (no arbitrary code execution)
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
    """Recursively evaluate a restricted AST node for math expressions."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](
            _safe_eval_node(node.left),
            _safe_eval_node(node.right),
        )
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def _evaluate_math(expression: str) -> float:
    """Parse and evaluate a math expression safely via AST."""
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval_node(tree.body)


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations accurately."""
    try:
        result = _evaluate_math(expression)
        # Display integers without trailing .0 when appropriate
        if result == int(result):
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as exc:
        logger.warning("Calculator error for expression '%s': %s", expression, exc)
        return f"Error: Could not evaluate expression. {exc}"


def build_tools(retriever) -> list:
    """Create the document retriever and calculator tools for the agent."""
    document_retriever = create_retriever_tool(
        retriever,
        name="document_retriever",
        description="Search the uploaded PDF documents and retrieve relevant information.",
        response_format="content_and_artifact",
    )
    return [document_retriever, calculator]


# =============================================================================
# SECTION 8: AGENT FACTORY
# =============================================================================

def build_agent(tools: list):
    """Build a LangChain agent with Ollama LLM, tools, system prompt, and memory."""
    llm = ChatOllama(
        model=OLLAMA_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
    )
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    logger.info("Agent created with tools: %s", [t.name for t in tools])
    return agent


# =============================================================================
# SECTION 9: AGENT RESPONSE PARSING (for terminal output)
# =============================================================================

def _extract_text_content(content: Any) -> str:
    """Normalize AIMessage content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts)
    return str(content) if content else ""


def parse_agent_response(
    result: dict,
    prior_message_count: int = 0,
) -> dict[str, Any]:
    """
    Extract the assistant reply, tools used, and retrieved context
    from messages added during the current turn only.
    """
    all_messages = result.get("messages", [])
    messages = all_messages[prior_message_count:]

    tools_used: list[str] = []
    retrieved_context: list[str] = []
    assistant_reply = ""

    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.get("name", "unknown")
                if name not in tools_used:
                    tools_used.append(name)

        if isinstance(message, ToolMessage) and message.name == "document_retriever":
            artifact = getattr(message, "artifact", None)
            if artifact:
                for idx, doc in enumerate(artifact, start=1):
                    page = doc.metadata.get("page", "unknown")
                    source = doc.metadata.get("source", "unknown")
                    snippet = doc.page_content.strip().replace("\n", " ")
                    if len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    retrieved_context.append(
                        f"[chunk {idx}, page {page}] ({os.path.basename(source)}): {snippet}"
                    )
            elif message.content:
                retrieved_context.append(str(message.content).strip())

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            assistant_reply = _extract_text_content(message.content)
            if assistant_reply:
                break

    return {
        "reply": assistant_reply or "I could not generate a response.",
        "tools_used": tools_used,
        "retrieved_context": retrieved_context,
    }


def print_turn_output(parsed: dict[str, Any]) -> None:
    """Pretty-print tools used, retrieved context, and the assistant reply."""
    if parsed["tools_used"]:
        print(f"Tools Used: {', '.join(parsed['tools_used'])}")
    else:
        print("Tools Used: (none)")

    if parsed["retrieved_context"]:
        print("--- Retrieved Context ---")
        for line in parsed["retrieved_context"]:
            print(line)
        print("-------------------------")

    print(f"\nAssistant: {parsed['reply']}\n")


# =============================================================================
# SECTION 10: TERMINAL CLI INTERFACE
# =============================================================================

HELP_TEXT = """
Available commands:
  help     - Show this help message
  clear    - Reset conversation memory (start a fresh chat)
  exit     - End the session (also: quit)

Embeddings are saved in ./chroma_db and reused on the next run.
Replace sample.pdf and restart with --reindex to rebuild the index.
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agentic RAG — chat with sample.pdf using Ollama and LangChain.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force rebuild embeddings from sample.pdf",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args()


def run_chat_loop(agent, thread_id: str) -> str:
    """
    Interactive terminal chat loop with conversation memory.

    Returns the final thread_id (may change after a 'clear' command).
    """
    print("\n" + "=" * 60)
    print("Agentic RAG — Interactive Chat")
    print("Type your question, or 'help' for commands.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in ("exit", "quit"):
            print("Goodbye!")
            break
        if command == "help":
            print(HELP_TEXT)
            continue
        if command == "clear":
            thread_id = str(uuid.uuid4())
            print("Conversation cleared. Memory reset.\n")
            continue

        config = {"configurable": {"thread_id": thread_id}}

        try:
            logger.info("Invoking agent for query: %s", user_input)

            # Snapshot message count so we only display metadata from this turn
            prior_state = agent.get_state(config)
            prior_message_count = len(prior_state.values.get("messages", []))

            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            parsed = parse_agent_response(result, prior_message_count)
            print_turn_output(parsed)
        except Exception as exc:
            logger.exception("Agent invocation failed")
            print(
                f"\nError: Failed to get a response. {exc}\n"
                "Check that Ollama is running and models are available.\n"
            )

    return thread_id


# =============================================================================
# SECTION 11: MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point: load or build embeddings, then start chat."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    try:
        pdf_path = validate_pdf_path(PDF_PATH)
        check_ollama_connection()

        if args.reindex or needs_reindex(pdf_path):
            print(f"\nIndexing PDF: {pdf_path}")
        else:
            print(f"\nDocument: {pdf_path}")

        vectorstore = load_or_index_vectorstore(pdf_path, force_reindex=args.reindex)

        retriever = create_mmr_retriever(vectorstore)
        tools = build_tools(retriever)
        agent = build_agent(tools)

        thread_id = str(uuid.uuid4())
        print("Ready. Starting chat...\n")
        run_chat_loop(agent, thread_id)

    except AppError as exc:
        logger.error(str(exc))
        print(f"\nError: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
