import { useState } from "react";

interface InputPanelProps {
  inputText: string;
  setInputText: (text: string) => void;
  seed: string;
  setSeed: (seed: string) => void;
  maxNewTokens: string;
  setMaxNewTokens: (maxNewTokens: string) => void;
  onExplainClick: (ignoreSpecialTokens: boolean, disableThinking: boolean) => void;
  inferenceStatus?: 'idle' | 'running' | 'success' | 'error' | string;
  isConfigReady: boolean;
  activeAttributorId: string | null;
  activeWrapperName: string | null;
}

export default function InputPanel(props: InputPanelProps) {
  const isRunning = props.inferenceStatus === 'running';
  const wordCount = props.inputText.trim().split(/\s+/).filter((word: string) => word.length > 0).length;
  // DAAM filters special tokens at generation time (they can't be toggled after the fact,
  // unlike the text views), so default to ignoring them to avoid the attention sink.
  const [isSpecialTokensDisabled, setIsSpecialTokensDisabled] = useState(true);
  const [isThinkingDisabled, setIsThinkingDisabled] = useState(false);
  const showDisableThinkingToggle = props.activeWrapperName === "hf_text_generation";

  const isButtonDisabled = !props.isConfigReady || isRunning || wordCount === 0;

  const handleSeedChange = (value: string) => {
    // Keep only digits so the value is always a valid non-negative integer (or empty).
    props.setSeed(value.replace(/[^0-9]/g, ""));
  };

  const randomizeSeed = () => {
    props.setSeed(String(Math.floor(Math.random() * 2 ** 31)));
  };

  const handleMaxNewTokensChange = (value: string) => {
    props.setMaxNewTokens(value.replace(/[^0-9]/g, "").replace(/^0+/, ""));
  };

  return (
    <div className="mt-5">

      <div className="flex flex-col gap-2">

        {/* TEXT AREA (disabled only during inference) */}
        <textarea
          className="bg-fill w-full min-h-48 p-4 text-fg font-mono text-sm outline-none disabled:opacity-50 resize-y"
          name="inputText"
          id="inputText"
          value={props.inputText}
          onChange={(e) => props.setInputText(e.target.value)}
          disabled={isRunning}
          placeholder="Type your prompt here..."
        ></textarea>

        {/* WORD COUNTER */}
        <div className="text-fg-faint font-mono font-medium text-sm text-center mt-1">
          Total words: {wordCount}
        </div>

        {/* SPECIAL TOKENS TOGGLE */}
        {props.activeAttributorId === "daam" && (
          <div className="mt-6 bg-fill text-fg-subtle font-mono text-xs font-medium uppercase flex justify-between p-4 ">
            <div>{"// DAAM will "}<span className="text-warn">{isSpecialTokensDisabled ? "ignore" : "consider"}</span> special tokens.</div>
            <button
              className="underline underline-offset-4 cursor-pointer hover:text-fg transition-colors"
              onClick={() => setIsSpecialTokensDisabled(!isSpecialTokensDisabled)}
            >
              {isSpecialTokensDisabled ? "Consider" : "Ignore"} Special Tokens
            </button>
          </div>
        )}

        {/* THINKING TOGGLE (only meaningful for reasoning-capable text generation models) */}
        {showDisableThinkingToggle && (
          <div className="mt-6 bg-fill text-fg-subtle font-mono text-xs font-medium uppercase flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div>{"// Thinking "}<span className="text-warn">{isThinkingDisabled ? "disabled" : "enabled"}</span>.</div>
            <button
              type="button"
              className="underline underline-offset-4 cursor-pointer hover:text-fg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={() => setIsThinkingDisabled(!isThinkingDisabled)}
              disabled={isRunning}
              title="For supported chat templates, requests non-thinking mode during generation."
            >
              {isThinkingDisabled ? "Enable" : "Disable"} Thinking
            </button>
          </div>
        )}

        {/* MAX NEW TOKENS (only meaningful for text generation) */}
        {showDisableThinkingToggle && (
          <div className="mt-2 bg-fill text-fg-subtle font-mono text-xs font-medium uppercase flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div>{"// Max new tokens "}<span className="text-warn">{props.maxNewTokens.trim() === "" ? "(default)" : props.maxNewTokens}</span></div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                inputMode="numeric"
                value={props.maxNewTokens}
                onChange={(e) => handleMaxNewTokensChange(e.target.value)}
                disabled={isRunning}
                placeholder="default"
                className="w-28 bg-sunken text-fg text-right px-2 py-1 outline-none border border-border focus:border-border-strong disabled:opacity-50"
              />
              <button
                type="button"
                onClick={() => props.setMaxNewTokens("")}
                disabled={isRunning || props.maxNewTokens.trim() === ""}
                title="Use backend default"
                className="underline underline-offset-4 cursor-pointer hover:text-fg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Clear
              </button>
            </div>
          </div>
        )}

        {/* SEED PICKER (only meaningful for image generation) */}
        {props.activeAttributorId === "daam" && (
          <div className="mt-2 bg-fill text-fg-subtle font-mono text-xs font-medium uppercase flex items-center justify-between gap-3 p-4">
            <div className="whitespace-nowrap">
              {"// Seed "}
              <span className="text-warn">
                {props.seed.trim() === "" ? "(random)" : "(fixed)"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                inputMode="numeric"
                value={props.seed}
                onChange={(e) => handleSeedChange(e.target.value)}
                disabled={isRunning}
                placeholder="random"
                className="w-32 bg-sunken text-fg text-right px-2 py-1 outline-none border border-border focus:border-border-strong disabled:opacity-50"
              />
              <button
                type="button"
                onClick={randomizeSeed}
                disabled={isRunning}
                title="Generate a random seed"
                className="cursor-pointer hover:text-fg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <i className='bx bx-dice-5 text-lg'></i>
              </button>
              <button
                type="button"
                onClick={() => props.setSeed("")}
                disabled={isRunning || props.seed.trim() === ""}
                title="Clear seed (use random)"
                className="underline underline-offset-4 cursor-pointer hover:text-fg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                Clear
              </button>
            </div>
          </div>
        )}

        {/* RUN BUTTON */}
        <div className="mt-0">
          <button
            className="bg-info-soft hover:bg-info-hover border border-info-line text-info w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
            onClick={() => props.onExplainClick(isSpecialTokensDisabled, showDisableThinkingToggle && isThinkingDisabled)}
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
