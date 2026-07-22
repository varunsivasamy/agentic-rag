"""
Agentic RAG App for Hugging Face Spaces (Gradio SDK).

Uses Hugging Face models directly for cloud deployment.
"""

import os
import sys
import uuid
import tempfile
import hashlib
from typing import Any, List, Optional

import gradio as gr
from huggingface_hub import InferenceClient

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.tools import tool
import ast
import operator

# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

# Fixed employee handbook PDF
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "Company_Employee_Handbook.pdf")

# Hugging Face models for cloud deployment
LLM_MODEL_NAME = "google/gemma-2-2b-it"
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

        # Build vector store (in-memory for HF Spaces)
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

        # Create agent using ChatHuggingFace
        llm = ChatHuggingFace(
            llm=self._create_llm(),
            tokenizer_name=LLM_MODEL_NAME,
        )

        checkpointer = InMemorySaver()
        self.agent = create_agent(
            model=llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt(),
            checkpointer=checkpointer,
        )

        self.pdf_filename = pdf_filename
        return len(chunks)

    def _create_llm(self):
        """Create Hugging Face LLM for agent."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        return model

    def _get_system_prompt(self) -> str:
        return """You are an intelligent HR assistant for the company employee handbook.

The Company Employee Handbook is ALREADY indexed and available. Do NOT ask the user to upload anything.

You have access to two tools:

1. document_retriever — searches the employee handbook for relevant policies and information
2. calculator — performs math calculations (e.g. leave balances, salary calculations)

Rules:

* For ANY question about policies, benefits, leave, roles, or company rules, call document_retriever FIRST.
* Use calculator for any numerical computations.
* Never hallucinate. Only answer using retrieved handbook content.
* If information is not found in the handbook, clearly say so.
* Keep responses concise, accurate, and professional."""

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


# Global RAG engine instance - auto-load employee handbook on startup
rag_engine = None

def initialize_rag():
    """Initialize RAG engine with the employee handbook on startup."""
    global rag_engine
    try:
        rag = RAGEngine()
        rag.index_pdf(PDF_PATH, "Company_Employee_Handbook.pdf")
        rag_engine = rag
        print("✅ Employee Handbook loaded and indexed successfully.")
    except Exception as exc:
        print(f"Error loading handbook: {exc}")


def chat(message, history):
    """Handle chat interaction."""
    global rag_engine

    if rag_engine is None:
        return "Please upload a PDF document first."

    try:
        response = rag_engine.query(message)
        return response["reply"]
    except Exception as exc:
        return f"Error: {exc}"


def clear_conversation():
    """Clear conversation memory."""
    global rag_engine
    if rag_engine:
        rag_engine.clear_memory()
    return []


# =============================================================================
# GRADIO UI
# =============================================================================

with gr.Blocks(theme="soft", title="Employee Handbook Assistant") as demo:
    gr.Markdown("# 🏢 Employee Handbook Assistant")
    gr.Markdown("Ask questions about company policies, benefits, leave, roles, and more — powered by the Company Employee Handbook.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## About")
            gr.Markdown("""
            - **Private data extraction** — Queries internal company handbook
            - **Agentic workflow** — LLM decides when to retrieve vs calculate
            - **No hallucination** — Answers grounded in handbook content only
            """)
            clear_btn = gr.Button("Clear conversation", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("## Chat")
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Ask about company policies...", placeholder="e.g. What is the leave policy?")
            send_btn = gr.Button("Send", variant="primary")

    chat_history = gr.State([])

    send_btn.click(
        fn=chat,
        inputs=[msg, chat_history],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    clear_btn.click(
        fn=clear_conversation,
        outputs=chatbot
    )

    gr.Examples(
        examples=[
            "What is the leave policy?",
            "What are the employee benefits?",
            "What is the code of conduct?",
            "How many days of annual leave do employees get?",
        ],
        inputs=msg
    )


if __name__ == "__main__":
    initialize_rag()
    demo.launch()
