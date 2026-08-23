const SUGGESTIONS = [
  "What is the leave policy?",
  "What are the employee benefits?",
  "What is the code of conduct?",
  "How many annual leave days do I get?",
  "What is the IT security policy?",
  "Explain the travel policy",
  "What is the company's branding guideline?",
  "What are the health and safety rules?",
];

interface Props {
  onSuggestion: (text: string) => void;
}

export default function WelcomeScreen({ onSuggestion }: Props) {
  return (
    <div className="welcome">
      <h2>How can I help you today?</h2>
      <p>I have access to all company documents. Ask me anything.</p>
      <div className="suggestions">
        {SUGGESTIONS.map((s) => (
          <button key={s} className="suggestion" onClick={() => onSuggestion(s)}>
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
