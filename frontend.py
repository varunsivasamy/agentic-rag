"""
Streamlit Frontend for Agentic RAG Application.

Web interface for chatting with PDF documents using Ollama and LangChain.
"""

import os
import uuid
import tempfile
import hashlib
from typing import Any

import streamlit as st

# Import RAG logic from app.py
from app import (
    index_pdf,
    load_or_index_vectorstore,
    create_mmr_retriever,
    build_tools,
    build_agent,
    validate_pdf_path,
    check_ollama_connection,
    OLLAMA_LLM_MODEL,
    OLLAMA_EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MMR_K,
    MMR_FETCH_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    SYSTEM_PROMPT,
    PDF_PATH,
)


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Agentic RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .message-user {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .message-assistant {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .tool-call {
        background-color: #fff3e0;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff9800;
    }
    .context-box {
        background-color: #e8f5e9;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        font-family: monospace;
    }
    .stToast {
        background-color: #fff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_hash" not in st.session_state:
        st.session_state.pdf_hash = None
    if "ollama_ready" not in st.session_state:
        st.session_state.ollama_ready = False
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "tools" not in st.session_state:
        st.session_state.tools = None


# =============================================================================
# RAG PIPELINE MANAGEMENT
# =============================================================================

def initialize_rag_pipeline(pdf_path: str):
    """Initialize or reload the RAG pipeline with a PDF."""
    try:
        # Check Ollama connection
        check_ollama_connection()
        st.session_state.ollama_ready = True

        # Load or build vector store
        vectorstore = load_or_index_vectorstore(pdf_path, force_reindex=False)

        # Create retriever and tools
        retriever = create_mmr_retriever(vectorstore)
        tools = build_tools(retriever)

        # Build agent
        agent = build_agent(tools)

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever = retriever
        st.session_state.tools = tools
        st.session_state.agent = agent
        st.session_state.thread_id = str(uuid.uuid4())

        return True
    except Exception as exc:
        st.error(f"Failed to initialize RAG pipeline: {exc}")
        return False


def get_pdf_hash(file) -> str:
    """Calculate hash of uploaded PDF file."""
    file.seek(0)
    return hashlib.md5(file.read()).hexdigest()


# =============================================================================
# MESSAGE PARSING
# =============================================================================

def _extract_text_content(content: Any) -> str:
    """Normalize message content to a plain string."""
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
    """Extract assistant reply, tools used, and retrieved context."""
    all_messages = result.get("messages", [])
    messages = all_messages[prior_message_count:]

    tools_used: list[str] = []
    retrieved_context: list[str] = []
    assistant_reply = ""

    for message in messages:
        if isinstance(message, type(st.session_state.agent).__annotations__.get("model", None).__class__.__bases__[0]) and hasattr(message, 'tool_calls'):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    name = tool_call.get("name", "unknown")
                    if name not in tools_used:
                        tools_used.append(name)

        if hasattr(message, 'name') and message.name == "document_retriever" and hasattr(message, 'content'):
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
        if isinstance(message, type(st.session_state.agent).__annotations__.get("model", None).__class__.__bases__[0]) and hasattr(message, 'content'):
            if hasattr(message, 'content') and message.content:
                assistant_reply = _extract_text_content(message.content)
                if assistant_reply:
                    break

    return {
        "reply": assistant_reply or "I could not generate a response.",
        "tools_used": tools_used,
        "retrieved_context": retrieved_context,
    }


# =============================================================================
# STREAMLIT UI
# =============================================================================

def render_chat_message(role: str, content: str, tools: list = None, context: list = None):
    """Render a chat message with optional tool calls and context."""
    css_class = "message-user" if role == "user" else "message-assistant"

    with st.container():
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
        st.write(content)
        st.markdown("</div>", unsafe_allow_html=True)

        # Show tool calls if any
        if tools:
            for tool_name in tools:
                st.markdown(f'<div class="tool-call">🛠️ Tool used: <strong>{tool_name}</strong></div>', unsafe_allow_html=True)

        # Show retrieved context if any
        if context:
            st.markdown('<div class="context-box"><strong>📚 Retrieved Context:</strong></div>', unsafe_allow_html=True)
            for ctx in context:
                st.text(ctx)


def main():
    """Main Streamlit application entry point."""
    init_session_state()

    st.markdown('<div class="main-header">📚 Agentic RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Chat with your PDF documents using an agentic RAG pipeline</div>', unsafe_allow_html=True)

    # Sidebar for configuration and upload
    with st.sidebar:
        st.header("⚙️ Configuration")

        # PDF Upload Section
        st.subheader("📄 Document")
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            help="Upload your PDF document to enable Q&A"
        )

        if uploaded_file is not None:
            current_hash = get_pdf_hash(uploaded_file)

            # Only reindex if it's a new file
            if current_hash != st.session_state.pdf_hash:
                with st.spinner("Processing PDF and building embeddings..."):
                    try:
                        # Save uploaded file to temp location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        # Initialize RAG pipeline
                        if initialize_rag_pipeline(temp_path):
                            st.session_state.pdf_hash = current_hash
                            st.session_state.messages = []
                            st.success(f"✅ {uploaded_file.name} loaded successfully!")
                            st.info(f"Using model: `{OLLAMA_LLM_MODEL}` | Embeddings: `{OLLAMA_EMBED_MODEL}`")
                        else:
                            st.error("Failed to load document. Check Ollama connection.")

                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except Exception as exc:
                        st.error(f"Error processing PDF: {exc}")

        # Settings
        st.subheader("⚙️ Settings")
        show_tool_calls = st.toggle("Show tool calls", value=True, help="Display which tools the agent used")
        show_context = st.toggle("Show retrieved context", value=True, help="Display document chunks used for answers")

        # Clear conversation
        if st.session_state.messages:
            if st.button("🗑️ Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

        # Info section
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        - **Agentic workflow** — LLM decides when to检索 vs calculate
        - **Local-first** — Runs entirely on your machine
        - **Ollama** — Uses `qwen2.5:1.5b` for chat, `nomic-embed-text` for embeddings
        """)

        # Show Ollama status
        if st.session_state.ollama_ready:
            st.success("🔌 Ollama connected")
        else:
            st.warning("🔌 Ollama not connected or models not pulled")

    # Main chat area
    st.divider()

    # Display existing messages
    for msg in st.session_state.messages:
        render_chat_message(
            msg["role"],
            msg["content"],
            msg.get("tools") if show_tool_calls else None,
            msg.get("context") if show_context else None
        )

    # Chat input
    if not st.session_state.agent:
        if not st.session_state.ollama_ready:
            st.info("Connect to Ollama and upload a PDF to start chatting.")
        else:
            st.info("Upload a PDF document to begin.")

    else:
        if prompt := st.chat_input("Ask a question about your document...", disabled=not st.session_state.agent):
            # Add user message to state
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
            })

            # Display user message
            render_chat_message("user", prompt)

            # Call agent
            with st.spinner("Thinking..."):
                try:
                    prior_state = st.session_state.agent.get_state(
                        {"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    prior_message_count = len(prior_state.values.get("messages", []))

                    result = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": prompt}]},
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )

                    parsed = parse_agent_response(result, prior_message_count)

                    # Add assistant message to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": parsed["reply"],
                        "tools": parsed["tools_used"] if show_tool_calls else None,
                        "context": parsed["retrieved_context"] if show_context else None,
                    })

                    # Display assistant message
                    render_chat_message(
                        "assistant",
                        parsed["reply"],
                        parsed["tools_used"] if show_tool_calls else None,
                        parsed["retrieved_context"] if show_context else None
                    )

                except Exception as exc:
                    st.error(f"Error generating response: {exc}")
                    st.exception(exc)


if __name__ == "__main__":
    main()
