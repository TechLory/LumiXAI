
interface TokenExplainedProps {
  token: string;
  score: number;
}

export default function TokenExplained({ token, score }: TokenExplainedProps) {

  const parsed_token = token.replace('Ġ', ' ').replace('##', '');

  const getColorClass = (s: number) => {
    if (s > 0) {
      if (s >= 0.9) return 'bg-green-950 text-white';
      if (s >= 0.8) return 'bg-green-800 text-white';
      if (s >= 0.7) return 'bg-green-700 text-white';
      if (s >= 0.6) return 'bg-green-600 text-white';
      if (s >= 0.5) return 'bg-green-500 text-white';
      if (s >= 0.4) return 'bg-green-400 text-black';
      if (s >= 0.3) return 'bg-green-300 text-black';
      if (s >= 0.2) return 'bg-green-200 text-black';
      if (s >= 0.1) return 'bg-green-100 text-black';
      return 'bg-green-50 text-black';               
    }
    else if (s < 0) {
      if (s <= -0.9) return 'bg-red-950 text-white';
      if (s <= -0.8) return 'bg-red-800 text-white';
      if (s <= -0.7) return 'bg-red-700 text-white';
      if (s <= -0.6) return 'bg-red-600 text-white';
      if (s <= -0.5) return 'bg-red-500 text-white';
      if (s <= -0.4) return 'bg-red-400 text-black';
      if (s <= -0.3) return 'bg-red-300 text-black';
      if (s <= -0.2) return 'bg-red-200 text-black';
      if (s <= -0.1) return 'bg-red-100 text-black';
      return 'bg-red-50 text-black';                
    }
    return 'bg-white text-neutral-500';
  };

  const colorClass = getColorClass(score);

  return (
    <div
      className={`inline-block p-1 px-2 rounded text-sm font-mono transition-colors ${colorClass}`}
      title={`Score: ${score.toFixed(5)}`}
    >
      {parsed_token}
    </div>
  );
}