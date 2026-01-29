import TokenExplained from "./TokenExplained";

interface OutputPanelProps {
  outputResult: {
    target_id: any;
    predicted_token?: string;
    tokens: string[];
    scores: any;
    generated_image?: string;
  };
}

export default function OutputPanel({ outputResult }: OutputPanelProps) {
  if (!outputResult) return null;

  // --- images ---
  if (outputResult.generated_image) {

    // Scores: list (Multi-token) / single string
    const heatmaps = Array.isArray(outputResult.scores) ? outputResult.scores : [outputResult.scores];

    return (
      <div className="mb-20 px-6">
        <div className="text-3xl font-semibold mb-6">Visual Explanation (DAAM)</div>

        {/* 1. Original Image (Large) */}
        <div className="flex justify-center mb-10">
          <div className="flex flex-col items-center p-4 bg-neutral-900">
            <span className="text-xs mb-2 uppercase font-bold">Generated Output</span>
            <img
              src={`data:image/png;base64,${outputResult.generated_image}`}
              alt="Generated Output"
              className="max-h-96 w-auto"
            />
          </div>
        </div>

        {/* 2. Heatmap Gallery (Token by Token) */}
        <h3 className="text-xl font-semibold mb-4">Per-Token Attribution</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">

          {outputResult.tokens.map((token, idx) => {
            const heatmapSrc = heatmaps[idx];
            if (!heatmapSrc) return null;

            return (
              <div key={idx} className="flex flex-col items-center p-2 bg-neutral-900 ">
                <span className="text-sm font-mono font-bold px-2 py-1 mb-2">
                  {token}
                </span>
                <img
                  src={`data:image/png;base64,${heatmapSrc}`}
                  alt={`Map for ${token}`}
                  className="rounded shadow-sm w-full h-auto"
                />
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // text mode
  return (
    <div className="mb-20 px-6">

      <div className="text-3xl font-semibold">Output (TEXT)</div>



      <div className="flex gap-1 mt-5">
        {outputResult.tokens?.map((token: string, index: number) => (
          <TokenExplained
            key={index}
            token={token}
            score={outputResult.scores?.[index]}
          />
        ))}
      </div>

      <div className="font-mono text-xl flex gap-2 mt-5">
        <div className="italic text-neutral-400">Model output: </div>
        {outputResult.target_id === 0 ? (
          <div className="font-bold">NEGATIVE</div>
        ) : outputResult.target_id === 1 ? (
          <div className="font-bold">POSITIVE</div>
        ) : (
          <div className="font-bold">{outputResult.predicted_token}</div>
        )}
      </div>

    </div>
  );
}

