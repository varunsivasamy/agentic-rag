"""
FastAPI backend for the Employee Handbook RAG chatbot.

Endpoints:
  GET  /              — Serve the frontend HTML
  POST /chat          — Send a message, get a response
  GET  /history/{id}  — Retrieve conversation history by session ID
  DELETE /history/{id}— Clear conversation history for a session
  GET  /status        — Health check / model info
"""

import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel

from rag_engine import load_and_index_pdfs, build_agent, query_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


# =============================================================================
# APP STATE
# =============================================================================

# Conversation history: { session_id: [ {role, content, sources, timestamp} ] }
conversation_store: dict[str, list[dict]] = defaultdict(list)

agent = None  # Shared agent instance (InMemorySaver handles per-thread memory)


# =============================================================================
# LIFESPAN — load PDFs and build agent on startup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Loading PDFs and building agent...")
    vectorstore = load_and_index_pdfs()
    agent = build_agent(vectorstore)
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down.")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Employee Handbook Assistant",
    description="Ask questions about company policies, benefits, and more.",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# =============================================================================
# SCHEMAS
# =============================================================================

class ChatRequest(BaseModel):
    session_id: str | None = None  # If None, a new session is created
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: list[str]
    tools_used: list[str]
    timestamp: str


class HistoryItem(BaseModel):
    role: str
    content: str
    sources: list[str] = []
    timestamp: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryItem]


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the chat frontend."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Handle a chat message and return the agent's response."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready. Try again shortly.")

    # Create new session if not provided
    session_id = req.session_id or str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Store user message
    conversation_store[session_id].append({
        "role": "user",
        "content": req.message,
        "sources": [],
        "timestamp": timestamp,
    })

    try:
        result = query_agent(agent, session_id, req.message)
    except Exception as exc:
        logger.exception("Agent error for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    reply_timestamp = datetime.utcnow().isoformat()

    # Store assistant response
    conversation_store[session_id].append({
        "role": "assistant",
        "content": result["reply"],
        "sources": result["sources"],
        "timestamp": reply_timestamp,
    })

    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        sources=result["sources"],
        tools_used=result["tools_used"],
        timestamp=reply_timestamp,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Return full conversation history for a session."""
    if session_id not in conversation_store:
        raise HTTPException(status_code=404, detail="Session not found.")
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryItem(**m) for m in conversation_store[session_id]],
    )


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    if session_id in conversation_store:
        del conversation_store[session_id]
    return {"message": f"History cleared for session {session_id}"}


@app.get("/status")
async def status():
    """Health check and model info."""
    return {
        "status": "ready" if agent is not None else "loading",
        "llm_model": "qwen2.5:1.5b",
        "embed_model": "nomic-embed-text",
        "pdf_folder": "./pdf",
        "active_sessions": len(conversation_store),
    }
