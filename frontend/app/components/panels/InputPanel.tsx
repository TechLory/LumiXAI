import { useState } from "react";

interface InputPanelProps {
  inputText: string;
  setInputText: (text: string) => void;
  onExplainClick: (ignoreSpecialTokens: boolean) => void;
  inferenceStatus?: 'idle' | 'running' | 'success' | 'error' | string;
  isConfigReady: boolean;
  activeAttributorId: string | null;
}

export default function InputPanel(props: InputPanelProps) {
  const isRunning = props.inferenceStatus === 'running';
  const wordCount = props.inputText.trim().split(/\s+/).filter((word: string) => word.length > 0).length;
  const [isSpecialTokensDisabled, setIsSpecialTokensDisabled] = useState(false); // DEBUG, PASSARE A BACKEND

  const isButtonDisabled = !props.isConfigReady || isRunning || wordCount === 0;

  return (
    <div className="mt-5">

      <div className="flex flex-col gap-2">

        {/* TEXT AREA (disabled only during inference) */}
        <textarea
          className="bg-neutral-600/30 w-full min-h-48 p-4 text-neutral-200 font-mono text-sm outline-none disabled:opacity-50 resize-y"
          name="inputText"
          id="inputText"
          value={props.inputText}
          onChange={(e) => props.setInputText(e.target.value)}
          disabled={isRunning}
          placeholder="Type your prompt here..."
        ></textarea>

        {/* WORD COUNTER */}
        <div className="text-neutral-500 font-mono font-medium text-sm text-center mt-1">
          Total words: {wordCount}
        </div>

        {/* SPECIAL TOKENS TOGGLE */}
        {props.activeAttributorId === "daam" && (
          <div className="mt-6 bg-neutral-600/30 text-neutral-400 font-mono text-xs font-medium uppercase flex justify-between p-4 ">
            <div>// DAAM will <span className="text-yellow-600">{isSpecialTokensDisabled ? "ignore" : "consider"}</span> special tokens.</div>
            <button
              className="underline underline-offset-4 cursor-pointer hover:text-neutral-300 transition-colors"
              onClick={() => setIsSpecialTokensDisabled(!isSpecialTokensDisabled)}
            >
              {isSpecialTokensDisabled ? "Consider" : "Ignore"} Special Tokens
            </button>
          </div>
        )}

        {/* RUN BUTTON */}
        <div className="mt-0">
          <button
            className="bg-blue-900/40 hover:bg-blue-800/60 border border-blue-700/50 text-blue-400 w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
            onClick={() => props.onExplainClick(isSpecialTokensDisabled)}
            disabled={isButtonDisabled}
          >
            {!props.isConfigReady ? (
              <><i className='bx bx-lock-alt text-lg'></i> Waiting for Configuration</>
            ) : isRunning ? (
              <><i className='bx bx-loader animate-spin text-lg'></i> Running Inference...</>
            ) : (
              <><i className='bx bx-play text-lg'></i> Run and Explain</>
            )}
          </button>
        </div>

      </div>
    </div>
  );
}
