export type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
  tools_used?: string[];
  timestamp: string;
};

export type ChatResponse = {
  session_id: string;
  reply: string;
  sources: string[];
  tools_used: string[];
  timestamp: string;
};
