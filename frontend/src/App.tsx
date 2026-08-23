import { useState, useEffect, useRef } from "react";
import "./App.css";
import type { Message, ChatResponse } from "./types";
import Header from "./components/Header";
import MessageBubble from "./components/MessageBubble";
import TypingIndicator from "./components/TypingIndicator";
import WelcomeScreen from "./components/WelcomeScreen";
import InputBar from "./components/InputBar";

const API = "http://localhost:8000";

export default function App() {
  const [messages, setMessages]   = useState<Message[]>([]);
  const [input, setInput]         = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading]     = useState(false);
  const [ready, setReady]         = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Poll backend until agent is ready
  useEffect(() => {
    const check = async () => {
      try {
        const res  = await fetch(`${API}/status`);
        const data = await res.json();
        data.status === "ready" ? setReady(true) : setTimeout(check, 2000);
      } catch {
        setTimeout(check, 3000);
      }
    };
    check();
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (text?: string) => {
    const message = (text ?? input).trim();
    if (!message || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: message, timestamp: new Date().toISOString() }]);
    setLoading(true);

    try {
      const res  = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message }),
      });
      const data: ChatResponse = await res.json();

      if (!res.ok) throw new Error((data as any).detail || "Server error");

      setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.reply,
          sources: data.sources,
          tools_used: data.tools_used,
          timestamp: data.timestamp,
        },
      ]);
    } catch (err: any) {
      setMessages((prev) => [...prev, {
        role: "assistant",
        content: `Error: ${err.message}`,
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  const clearChat = async () => {
    if (sessionId) {
      await fetch(`${API}/history/${sessionId}`, { method: "DELETE" });
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
