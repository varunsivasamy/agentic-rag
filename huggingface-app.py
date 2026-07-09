"""
Hugging Face Spaces App for Agentic RAG.

This version uses Hugging Face models instead of Ollama for cloud deployment.
"""

import os
import sys
import uuid
import tempfile
import hashlib
from typing import Any, List, Optional

import streamlit as st
from huggingface_hub import InferenceClient

# Try to import HF integrations
try:
    from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.tools import create_retriever_tool
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.checkpoint.memory import InMemorySaver
    from langchain.agents import create_agent
    from langchain.tools import tool
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import ast
import operator

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

# Hugging Face models for cloud deployment
LLM_MODEL_NAME = "google/gemma-2-2b-it"  # Lightweight, good for inference
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================================
# SAFE CALCULATOR TOOL
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
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def _evaluate_math(expression: str) -> float:
    tree = ast.parse(expression.strip(), mode="eval")
    return _safe_eval_node(tree.body)


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


# =============================================================================
# RAG ENGINE
# =============================================================================

class RAGEngine:
    """RAG engine using Hugging Face models."""

    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.tools = None
        self.agent = None
        self.thread_id = None
        self.embeddings = None
        self.pdf_filename = None

    def index_pdf(self, pdf_path: str, pdf_filename: str):
        """Index a PDF document."""
        # Load and split PDF
        documents = PyPDFLoader(pdf_path).load()
        non_empty_pages = [doc for doc in documents if doc.page_content and doc.page_content.strip()]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)

        # Create embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

        # Build vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20},
        )

        # Create tools
        retriever_tool = create_retriever_tool(
            self.retriever,
            name="document_retriever",
            description="Search the uploaded PDF documents and retrieve relevant information.",
            response_format="content_and_artifact",
        )
        self.tools = [retriever_tool, calculator]

        # Create agent
        llm_client = InferenceClient(model=LLM_MODEL_NAME)
        self.agent = self._create_agent(self.tools)

        self.pdf_filename = pdf_filename
        return len(chunks)

    def _create_agent(self, tools: List):
        """Create a LangChain agent."""
        checkpointer = InMemorySaver()
        agent = create_agent(
            model=None,  # Will be set later
            tools=tools,
            system_prompt=self._get_system_prompt(),
            checkpointer=checkpointer,
        )
        return agent

    def _get_system_prompt(self) -> str:
        return """You are an intelligent document assistant.

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

    def query(self, question: str) -> dict:
        """Query the agent with a question."""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call index_pdf first.")

        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": self.thread_id}}

        prior_state = self.agent.get_state(config)
        prior_message_count = len(prior_state.values.get("messages", []))

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        return self._parse_response(result, prior_message_count)

    def _parse_response(self, result: dict, prior_message_count: int) -> dict:
        """Parse agent response."""
        all_messages = result.get("messages", [])
        messages = all_messages[prior_message_count:]

        tools_used = []
        retrieved_context = []
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

        return {
            "reply": assistant_reply or "I could not generate a response.",
            "tools_used": tools_used,
            "retrieved_context": retrieved_context,
        }

    def clear_memory(self):
        """Clear conversation memory."""
        self.thread_id = None


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 2rem; font-weight: bold; color: #1a1a2e; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .message-user { background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .message-assistant { background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .tool-call { background-color: #fff3e0; padding: 0.75rem; border-radius: 0.375rem; margin: 0.5rem 0; border-left: 4px solid #ff9800; }
    .context-box { background-color: #e8f5e9; padding: 0.75rem; border-radius: 0.375rem; margin: 0.5rem 0; font-size: 0.85rem; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None
    if "pdf_loaded" not in st.session_state:
        st.session_state.pdf_loaded = False


def render_chat_message(role: str, content: str, tools: list = None, context: list = None):
    """Render a chat message."""
    css_class = "message-user" if role == "user" else "message-assistant"

    with st.container():
        st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

        if tools:
            for tool_name in tools:
                st.markdown(f'<div class="tool-call">Tool used: {tool_name}</div>', unsafe_allow_html=True)

        if context:
            st.markdown('<div class="context-box"><strong>Retrieved Context:</strong></div>', unsafe_allow_html=True)
            for ctx in context:
                st.text(ctx)


def main():
    """Main Streamlit application."""
    init_session_state()

    st.markdown('<div class="main-header">Agentic RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chat with your PDF documents using an agentic RAG pipeline on Hugging Face</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # PDF Upload
        st.subheader("Document")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            current_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

            if current_hash != st.session_state.pdf_hash:
                with st.spinner("Processing PDF and building embeddings (using Hugging Face models)..."):
                    try:
                        # Save uploaded file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        # Initialize RAG engine
                        rag = RAGEngine()
                        num_chunks = rag.index_pdf(temp_path, uploaded_file.name)

                        st.session_state.rag_engine = rag
                        st.session_state.pdf_hash = current_hash
                        st.session_state.pdf_loaded = True
                        st.success(f"✅ {uploaded_file.name} loaded! ({num_chunks} chunks)")
                        st.info(f"Using model: {LLM_MODEL_NAME} | Embeddings: {EMBED_MODEL_NAME}")

                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)

                    except Exception as exc:
                        st.error(f"Error processing PDF: {exc}")
                        import traceback
                        st.exception(exc)

        # Settings
        st.subheader("Settings")
        show_tool_calls = st.toggle("Show tool calls", value=True)
        show_context = st.toggle("Show retrieved context", value=True)

        # Clear conversation
        if st.session_state.messages:
            if st.button("Clear conversation"):
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_memory()
                st.session_state.messages = []
                st.rerun()

        # Info
        st.divider()
        st.subheader("About")
        st.markdown("""
        - **Agentic workflow** — LLM decides when to retrieve vs calculate
        - **Cloud-based** — Runs on Hugging Face Spaces
        - **Models** — Uses Hugging Face inference API
        """)

    # Main chat area
    st.divider()

    # Display messages
    for msg in st.session_state.messages:
        render_chat_message(
            msg["role"],
            msg["content"],
            msg.get("tools") if show_tool_calls else None,
            msg.get("context") if show_context else None
        )

    # Chat input
    if not st.session_state.pdf_loaded:
        st.info("Upload a PDF document to begin chatting.")

    else:
        if prompt := st.chat_input("Ask a question about your document...", disabled=not st.session_state.pdf_loaded):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            render_chat_message("user", prompt)

            # Call agent
            with st.spinner("Thinking..."):
                try:
                    rag = st.session_state.rag_engine
                    response = rag.query(prompt)

                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["reply"],
                        "tools": response["tools_used"] if show_tool_calls else None,
                        "context": response["retrieved_context"] if show_context else None,
                    })

                    render_chat_message(
                        "assistant",
                        response["reply"],
                        response["tools_used"] if show_tool_calls else None,
                        response["retrieved_context"] if show_context else None
                    )

                except Exception as exc:
                    st.error(f"Error generating response: {exc}")
                    import traceback
                    st.exception(exc)


if __name__ == "__main__":
    main()
