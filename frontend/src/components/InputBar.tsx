import { useRef } from "react";

interface Props {
  value: string;
  onChange: (val: string) => void;
  onSend: () => void;
  onClear: () => void;
  disabled: boolean;
}

export default function InputBar({ value, onChange, onSend, onClear, disabled }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    // Auto resize
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 120) + "px";
    }
  };

  return (
    <footer className="input-area">
      <button className="clear-btn" onClick={onClear}>Clear</button>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKey}
        placeholder="Ask anything about the company..."
        rows={1}
        disabled={disabled}
      />
      <button
        className="send-btn"
        onClick={onSend}
        disabled={disabled || !value.trim()}
      >
        Send
      </button>
    </footer>
  );
}
