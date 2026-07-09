"""
Agentic RAG Engine — Core logic for PDF processing, embedding, and agent operations.

This module provides a modular RAG engine that can work with:
- Local Ollama (for local development)
- Hugging Face models (for cloud deployment)
"""

import os
import uuid
import hashlib
import shutil
import json
import tempfile
from typing import Optional, List, Any, Dict
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

# Try to import HF integrations (may not be available in all environments)
try:
    from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = "./chroma_db"
INDEX_META_FILE = os.path.join(CHROMA_DIR, "index_meta.json")
COLLECTION_NAME = "pdf_documents"

# Default models (can be overridden)
DEFAULT_LLM_MODEL = "qwen2.5:1.5b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PDFMetadata:
    """Metadata about a processed PDF document."""
    filename: str
    file_hash: str
    num_pages: int
    num_chunks: int
    uploaded_at: str


@dataclass
class AgentResponse:
    """Response from the RAG agent."""
    reply: str
    tools_used: List[str]
    retrieved_context: List[str]
    source_documents: List[Any]


# =============================================================================
# ERROR HANDLING
# =============================================================================

class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass


class OllamaNotAvailableError(RAGError):
    """Ollama is not available in this environment."""
    pass


class HFNotAvailableError(RAGError):
    """Hugging Face integrations are not available."""
    pass


# =============================================================================
# PDF PROCESSING
# =============================================================================

def get_pdf_hash(file_path: str) -> str:
    """Calculate SHA256 hash of a PDF file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_pdf_hash_bytes(file_bytes: bytes) -> str:
    """Calculate SHA256 hash of file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_pdf_path(pdf_path: str) -> str:
    """Validate that the PDF path exists and is a non-empty file."""
    resolved = os.path.abspath(pdf_path)

    if not os.path.isfile(resolved):
        raise RAGError(f"PDF file not found: {resolved}")

    if not resolved.lower().endswith(".pdf"):
        raise RAGError(f"Expected a .pdf file, got: {resolved}")

    if os.path.getsize(resolved) == 0:
        raise RAGError(f"PDF file is empty: {resolved}")

    return resolved


def process_pdf(pdf_path: str) -> List:
    """Load and split a PDF into text chunks."""
    try:
        documents = PyPDFLoader(pdf_path).load()
    except Exception as exc:
        raise RAGError(f"Failed to read PDF: {exc}") from exc

    if not documents:
        raise RAGError("The PDF contains no pages.")

    non_empty_pages = [
        doc for doc in documents if doc.page_content and doc.page_content.strip()
    ]
    if not non_empty_pages:
        raise RAGError("The PDF contains no extractable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    try:
        chunks = splitter.split_documents(documents)
    except Exception as exc:
        raise RAGError(f"Failed to split PDF text: {exc}") from exc

    if not chunks:
        raise RAGError("Text splitting produced zero chunks.")

    return chunks


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def create_local_embeddings(model_name: str = None):
    """Create Ollama embeddings."""
    if not OLLAMA_AVAILABLE:
        raise OllamaNotAvailableError(
            "Ollama is not available. Install with: pip install langchain-ollama"
        )
    model = model_name or DEFAULT_EMBED_MODEL
    return OllamaEmbeddings(model=model)


def create_hf_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Create Hugging Face embeddings."""
    if not HF_AVAILABLE:
        raise HFNotAvailableError(
            "Hugging Face integrations not available. Install with: "
            "pip install langchain-huggingface"
        )
    return HuggingFaceEmbeddings(model_name=model_name)


# =============================================================================
# CHROMA DB OPERATIONS
# =============================================================================

def _pdf_fingerprint(pdf_path: str, embed_model: str, chunk_size: int, chunk_overlap: int) -> dict:
    """Build a fingerprint of the PDF file for cache invalidation."""
    stat = os.stat(pdf_path)
    return {
        "pdf_path": os.path.abspath(pdf_path),
        "pdf_mtime": stat.st_mtime,
        "pdf_size": stat.st_size,
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _read_index_meta() -> Optional[dict]:
    """Read saved index metadata if it exists."""
    if not os.path.isfile(INDEX_META_FILE):
        return None
    try:
        with open(INDEX_META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return None


def _write_index_meta(pdf_path: str, embed_model: str, chunk_size: int, chunk_overlap: int) -> None:
    """Persist metadata after a successful index build."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    with open(INDEX_META_FILE, "w", encoding="utf-8") as f:
        json.dump(_pdf_fingerprint(pdf_path, embed_model, chunk_size, chunk_overlap), f, indent=2)


def needs_reindex(pdf_path: str, embed_model: str, chunk_size: int, chunk_overlap: int) -> bool:
    """Return True if embeddings must be rebuilt from the PDF."""
    if not os.path.isdir(CHROMA_DIR):
        return True

    saved = _read_index_meta()
    if saved is None:
        return True

    return saved != _pdf_fingerprint(pdf_path, embed_model, chunk_size, chunk_overlap)


def build_vectorstore(chunks: List, embed_model_name: str = None, persist: bool = True) -> Chroma:
    """Build a ChromaDB vector store from document chunks."""
    try:
        if persist:
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
            os.makedirs(CHROMA_DIR, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=create_local_embeddings(embed_model_name),
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR if persist else None,
        )

        if persist:
            _write_index_meta(
                os.path.join(CHROMA_DIR, "placeholder.pdf"),
                embed_model_name or DEFAULT_EMBED_MODEL,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )

        return vectorstore
    except Exception as exc:
        if persist:
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        raise RAGError(f"Failed to build vector store: {exc}") from exc


def load_vectorstore(embed_model_name: str = None) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=create_local_embeddings(embed_model_name),
            persist_directory=CHROMA_DIR,
        )
        if vectorstore._collection.count() == 0:
            raise RAGError("ChromaDB collection is empty.")
        return vectorstore
    except RAGError:
        raise
    except Exception as exc:
        raise RAGError(
            f"Failed to load vector store from {CHROMA_DIR}: {exc}\n"
            "Run with reindex=True to rebuild embeddings."
        ) from exc


# =============================================================================
# MMR RETRIEVER
# =============================================================================

def create_mmr_retriever(
    vectorstore: Chroma,
    k: int = 5,
    fetch_k: int = 20,
    search_type: str = "mmr"
):
    """Create an MMR retriever with the configured search parameters."""
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )


# =============================================================================
# TOOLS
# =============================================================================

import operator
import ast


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate a restricted AST node for math expressions."""
    _SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

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


def create_calculator_tool():
    """Create a safe calculator tool."""

    @tool
    def calculator(expression: str) -> str:
        """Perform mathematical calculations accurately."""
        try:
            result = _evaluate_math(expression)
            if result == int(result):
                return str(int(result))
            return str(result)
        except ZeroDivisionError:
            return "Error: Division by zero."
        except Exception as exc:
            return f"Error: Could not evaluate expression. {exc}"

    return calculator


def create_retriever_tool_from_retriever(retriever, name: str = "document_retriever", description: str = None):
    """Create a document retriever tool from a retriever."""
    if description is None:
        description = "Search the uploaded PDF documents and retrieve relevant information."

    return create_retriever_tool(
        retriever,
        name=name,
        description=description,
        response_format="content_and_artifact",
    )


# =============================================================================
# AGENT
# =============================================================================

def create_agent_instance(tools: List, llm_model: str = None, system_prompt: str = None):
    """Build a LangChain agent with tools, model, and memory."""
    if not OLLAMA_AVAILABLE:
        raise OllamaNotAvailableError(
            "Ollama is not available. Install with: pip install langchain-ollama"
        )

    llm = ChatOllama(
        model=llm_model or DEFAULT_LLM_MODEL,
        temperature=0.2,
    )
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt or _DEFAULT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


_DEFAULT_SYSTEM_PROMPT = """You are an intelligent document assistant.

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
# RAG ENGINE (Main API)
# =============================================================================

class RAGEngine:
    """Main RAG engine class that orchestrates the entire pipeline."""

    def __init__(
        self,
        llm_model: str = None,
        embed_model: str = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        chroma_dir: str = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

        self.vectorstore = None
        self.retriever = None
        self.tools = None
        self.agent = None
        self.thread_id = None
        self.pdf_metadata = None

    def index_pdf(self, pdf_path: str) -> PDFMetadata:
        """Index a PDF document and build embeddings."""
        pdf_path = validate_pdf_path(pdf_path)

        # Process PDF
        chunks = process_pdf(pdf_path)

        # Build vector store
        self.vectorstore = build_vectorstore(chunks, self.embed_model)

        # Create retriever
        self.retriever = create_mmr_retriever(self.vectorstore)

        # Create tools
        retriever_tool = create_retriever_tool_from_retriever(
            self.retriever,
            description="Search the uploaded PDF documents and retrieve relevant information.",
        )
        calculator_tool = create_calculator_tool()
        self.tools = [retriever_tool, calculator_tool]

        # Build agent
        self.agent = create_agent_instance(
            self.tools,
            self.llm_model,
            _DEFAULT_SYSTEM_PROMPT,
        )

        # Save PDF metadata
        doc_count = len(chunks)
        self.pdf_metadata = PDFMetadata(
            filename=os.path.basename(pdf_path),
            file_hash=get_pdf_hash(pdf_path),
            num_pages=0,  # Would need to read PDF again to get this
            num_chunks=doc_count,
            uploaded_at=uuid.uuid4().hex[:8],
        )

        return self.pdf_metadata

    def index_pdf_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> PDFMetadata:
        """Index a PDF from bytes (for file upload scenarios)."""
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        try:
            result = self.index_pdf(temp_path)
            result.filename = filename
            result.file_hash = get_pdf_hash_bytes(pdf_bytes)
            return result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def query(self, question: str) -> AgentResponse:
        """Query the RAG agent with a question."""
        if self.agent is None:
            raise RAGError("Agent not initialized. Call index_pdf first.")

        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())

        prior_state = self.agent.get_state({"configurable": {"thread_id": self.thread_id}})
        prior_message_count = len(prior_state.values.get("messages", []))

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": self.thread_id}},
        )

        # Parse response
        tools_used = []
        retrieved_context = []
        assistant_reply = ""

        messages = result.get("messages", [])
        for message in messages[prior_message_count:]:
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

        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                if isinstance(message.content, str):
                    assistant_reply = message.content
                elif isinstance(message.content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in message.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    assistant_reply = "\n".join(text_parts)
                break

        return AgentResponse(
            reply=assistant_reply or "I could not generate a response.",
            tools_used=tools_used,
            retrieved_context=retrieved_context,
            source_documents=[],
        )

    def clear_memory(self):
        """Clear conversation memory."""
        self.thread_id = None

    def reset(self):
        """Reset the entire RAG engine."""
        self.vectorstore = None
        self.retriever = None
        self.tools = None
        self.agent = None
        self.thread_id = None
        self.pdf_metadata = None
        if os.path.isdir(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)


# =============================================================================
# HUGGING FACE MODE (for cloud deployment)
# =============================================================================

class HuggingFaceEngine:
    """RAG engine using Hugging Face models for cloud deployment."""

    def __init__(
        self,
        llm_model: str = "google/gemma-2-2b-it",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        if not HF_AVAILABLE:
            raise HFNotAvailableError(
                "Hugging Face integrations not available. Install with: "
                "pip install langchain-huggingface transformers"
            )

        self.llm_model = llm_model
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.vectorstore = None
        self.retriever = None
        self.tools = None
        self.agent = None
        self.thread_id = None
        self.pdf_metadata = None

    def index_pdf(self, pdf_path: str) -> PDFMetadata:
        """Index a PDF document using Hugging Face models."""
        pdf_path = validate_pdf_path(pdf_path)

        # Process PDF
        chunks = process_pdf(pdf_path)

        # Build vector store with HF embeddings
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=create_hf_embeddings(self.embed_model),
            collection_name=self.collection_name,
            persist_directory=None,  # In-memory for HF Spaces
        )

        # Create retriever
        self.retriever = create_mmr_retriever(self.vectorstore)

        # Create tools
        retriever_tool = create_retriever_tool_from_retriever(
            self.retriever,
            description="Search the uploaded PDF documents and retrieve relevant information.",
        )
        calculator_tool = create_calculator_tool()
        self.tools = [retriever_tool, calculator_tool]

        # Build agent with Hugging Face LLM
        llm = ChatHuggingFace(
            llm=self._create_hf_llm(),
            tokenizer=self._create_hf_tokenizer(),
        )
        checkpointer = InMemorySaver()
        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

        # Save PDF metadata
        self.pdf_metadata = PDFMetadata(
            filename=os.path.basename(pdf_path),
            file_hash=get_pdf_hash(pdf_path),
            num_pages=len(set(c.metadata.get("page", 0) for c in chunks)),
            num_chunks=len(chunks),
            uploaded_at=uuid.uuid4().hex[:8],
        )

        return self.pdf_metadata

    def _create_hf_llm(self):
        """Create Hugging Face LLM."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        return model

    def _create_hf_tokenizer(self):
        """Create Hugging Face tokenizer."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.llm_model)

    def query(self, question: str) -> AgentResponse:
        """Query the RAG agent with a question."""
        # Similar implementation as RAGEngine
        if self.agent is None:
            raise RAGError("Agent not initialized. Call index_pdf first.")

        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())

        # Simplified query for now
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": self.thread_id}},
        )

        # Parse response
        tools_used = []
        retrieved_context = []
        assistant_reply = ""

        messages = result.get("messages", [])
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

        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                if isinstance(message.content, str):
                    assistant_reply = message.content
                elif isinstance(message.content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in message.content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    assistant_reply = "\n".join(text_parts)
                break

        return AgentResponse(
            reply=assistant_reply or "I could not generate a response.",
            tools_used=tools_used,
            retrieved_context=retrieved_context,
            source_documents=[],
        )

    def clear_memory(self):
        """Clear conversation memory."""
        self.thread_id = None
