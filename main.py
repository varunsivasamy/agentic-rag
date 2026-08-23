"""
FastAPI backend — connects React frontend to the LangChain agent.
Conversation history is persisted in SQLite (chat_history.db).
"""

import json
import logging
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import load_and_index_pdfs, build_agent, query_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

DB_PATH = "chat_history.db"

# =============================================================================
# SQLite — conversation history store
# =============================================================================

def init_db():
    """Create the messages table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                sources   TEXT NOT NULL DEFAULT '[]',
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("SQLite DB ready at %s", DB_PATH)


def db_save_message(session_id: str, role: str, content: str,
                    sources: list[str], timestamp: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, sources, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources), timestamp),
        )
        conn.commit()


def db_get_history(session_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content, sources, timestamp FROM messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [
        {"role": r[0], "content": r[1],
         "sources": json.loads(r[2]), "timestamp": r[3]}
        for r in rows
    ]


def db_delete_history(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()


def db_active_sessions() -> int:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM messages"
        ).fetchone()
    return row[0] if row else 0


# =============================================================================
# APP STATE
# =============================================================================

agent_executor = None


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_executor
    init_db()
    logger.info("Loading PDFs and building agent...")
    vectorstore = load_and_index_pdfs()
    agent_executor = build_agent(vectorstore)
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down.")


# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="Company Assistant", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SCHEMAS
# =============================================================================

class ChatRequest(BaseModel):
    session_id: str | None = None
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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if agent_executor is None:
        raise HTTPException(status_code=503, detail="Agent not ready yet.")

    session_id = req.session_id or str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Save user message to DB
    db_save_message(session_id, "user", req.message, [], timestamp)

    # Load full history for context (exclude the message we just saved)
    history = db_get_history(session_id)[:-1]

    try:
        result = query_agent(
            agent=agent_executor,
            thread_id=session_id,
            question=req.message,
            history=history,
        )
    except Exception as exc:
        logger.exception("Agent error for session %s", session_id)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    reply_ts = datetime.utcnow().isoformat()

    # Save assistant reply to DB
    db_save_message(session_id, "assistant", result["reply"], result["sources"], reply_ts)

    return ChatResponse(
        session_id=session_id,
        reply=result["reply"],
        sources=result["sources"],
        tools_used=result["tools_used"],
        timestamp=reply_ts,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    messages = db_get_history(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found.")
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryItem(**m) for m in messages],
    )


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    db_delete_history(session_id)
    return {"message": f"Session {session_id} cleared."}


@app.get("/status")
async def status():
    return {
        "status": "ready" if agent_executor is not None else "loading",
        "llm_model": "qwen2.5:1.5b",
        "embed_model": "nomic-embed-text",
        "active_sessions": db_active_sessions(),
        "history_db": DB_PATH,
    }
