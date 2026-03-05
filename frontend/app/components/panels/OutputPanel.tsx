import TokenExplained from "../layout/TokenExplained";
import TextGenView from "../layout/TextGenView";
import ImageGenView from "../layout/ImageGenView";

export interface OutputResult {
  target_id?: string | number;
  predicted_token?: string;
  tokens?: string[];
  scores?: any; // array / matrix
  generated_image?: string;
}

interface OutputPanelProps {
  outputResult: OutputResult | null;
}

export default function OutputPanel({ outputResult }: OutputPanelProps) {

  if (!outputResult) return null; // todo: placeholder when no output is available

  // Dynamic Title
  let taskTitle = "Text Classification";
  if (outputResult.generated_image) taskTitle = "Image Generation";
  if (outputResult.target_id === "text_generation") taskTitle = "Text Generation";

  return (
    <div className="mt-5 mb-10">
      <div className="flex flex-col gap-0">

        <div className="uppercase p-2 text-sm bg-neutral-400/10 text-neutral-300 font-mono mb-2"><span className="font-bold mr-5">// TASK:</span>{taskTitle}</div>

        <div className="bg-neutral-400/10 p-6">

          {/* --- CASE 1: IMAGE GENERATION --- */}
          {outputResult.generated_image && (
            <ImageGenView
              baseImage={outputResult.generated_image}
              tokens={outputResult.tokens || []}
              heatmaps={Array.isArray(outputResult.scores) ? outputResult.scores : [outputResult.scores]}
            />
          )}

          {/* --- CASE 2: TEXT GENERATION --- */}
          {!outputResult.generated_image && outputResult.target_id === "text_generation" && (
            <TextGenView trace={outputResult.scores} />
          )}

          {/* --- CASE 3: TEXT CLASSIFICATION --- */}
          {!outputResult.generated_image && outputResult.target_id !== "text_generation" && (
            <div className="flex flex-col gap-6">

              {/* Token Box */}
              <div className="flex gap-2 flex-wrap p-5 bg-neutral-900/50 rounded-lg border border-neutral-700/50">
                {outputResult.tokens?.map((token: string, index: number) => (
                  <TokenExplained
                    key={index}
                    token={token}
                    score={outputResult.scores?.[index]}
                  />
                ))}
              </div>

              {/* Label Box */}
              <div className="font-mono text-lg flex gap-3 items-center justify-center p-4 bg-neutral-900/50 rounded-lg border border-neutral-700/50">
                <div className="uppercase text-neutral-500 text-sm tracking-wider">Predicted Class: </div>

                {outputResult.target_id === 0 ? (
                  <div className="font-bold text-red-400 bg-red-400/10 px-3 py-1 rounded">NEGATIVE</div>
                ) : outputResult.target_id === 1 ? (
                  <div className="font-bold text-emerald-400 bg-emerald-400/10 px-3 py-1 rounded">POSITIVE</div>
                ) : (
                  <div className="font-bold text-blue-400 bg-blue-400/10 px-3 py-1 rounded">
                    {outputResult.predicted_token || "UNKNOWN"}
                  </div>
                )}
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}