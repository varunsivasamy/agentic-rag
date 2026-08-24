import { useState, useEffect, useRef } from "react";
import "./App.css";
import type { Message, ChatResponse } from "./types";
import Header from "./components/Header";
import MessageBubble from "./components/MessageBubble";
import TypingIndicator from "./components/TypingIndicator";
import WelcomeScreen from "./components/WelcomeScreen";
import InputBar from "./components/InputBar";

// Vite proxy forwards /chat /history /status → FastAPI at localhost:8000
const API = "";

export default function App() {
  const [messages, setMessages]   = useState<Message[]>([]);
  const [input, setInput]         = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading]     = useState(false);
  const [ready, setReady]         = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Poll /status until backend agent is ready
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API}/status`);
        if (!res.ok) { if (!cancelled) setTimeout(check, 2000); return; }
        const data = await res.json();
        if (cancelled) return;
        data.status === "ready" ? setReady(true) : setTimeout(check, 2000);
      } catch {
        if (!cancelled) setTimeout(check, 3000);
      }
    };
    check();
    return () => { cancelled = true; };
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Safe JSON parse — never throws on empty/bad body
  const safeJson = async (res: Response): Promise<any> => {
    const text = await res.text();
    if (!text || !text.trim()) return {};
    try { return JSON.parse(text); }
    catch { return { detail: `Server returned invalid response (status ${res.status})` }; }
  };

  const sendMessage = async (text?: string) => {
    const message = (text ?? input).trim();
    if (!message || loading) return;

    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: message, timestamp: new Date().toISOString() },
    ]);
    setLoading(true);

    try {
      const res  = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message }),
      });

      const data = await safeJson(res);

      if (!res.ok) {
        throw new Error(data?.detail || `Server error (${res.status})`);
      }

      const chatData = data as ChatResponse;
      setSessionId(chatData.session_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: chatData.reply || "No response received.",
          sources:    chatData.sources    ?? [],
          tools_used: chatData.tools_used ?? [],
          timestamp:  chatData.timestamp  ?? new Date().toISOString(),
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `⚠️ ${err.message}`,
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = async () => {
    if (sessionId) {
      try {
        await fetch(`${API}/history/${sessionId}`, { method: "DELETE" });
      } catch { /* ignore */ }
      setSessionId(null);
    }
    setMessages([]);
  };

  return (
    <div className="app">
      <Header ready={ready} />

      <main className="messages">
        {messages.length === 0 ? (
          <WelcomeScreen onSuggestion={sendMessage} />
        ) : (
          messages.map((msg, i) => <MessageBubble key={i} message={msg} />)
        )}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </main>

      <InputBar
        value={input}
        onChange={setInput}
        onSend={sendMessage}
        onClear={clearChat}
        disabled={!ready || loading}
      />
    </div>
  );
}
