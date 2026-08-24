interface HeaderProps {
  ready: boolean;
}

export default function Header({ ready }: HeaderProps) {
  return (
    <header className="header">
      <div>
        <h1>🏢 Company Assistant</h1>
        <p>Ask anything about the company — policies, IT, travel, conduct, and more</p>
      </div>
      <div className="status">
        <span className={`dot ${ready ? "ready" : ""}`} />
        <span>{ready ? "Ready" : "Loading..."}</span>
      </div>
    </header>
  );
}
