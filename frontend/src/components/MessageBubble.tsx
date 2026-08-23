import { Message } from "../types";

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const { role, content, sources, tools_used } = message;

  return (
    <div className={`message ${role}`}>
      <div className="bubble">{content}</div>

      {sources && sources.length > 0 && (
        <div className="sources">
          {sources.map((s, i) => (
            <span key={i}>📄 {s}</span>
          ))}
        </div>
      )}

      {tools_used && tools_used.length > 0 && (
        <div className="tools-used">
          🔧 {tools_used.join(", ")}
        </div>
      )}
    </div>
  );
}
