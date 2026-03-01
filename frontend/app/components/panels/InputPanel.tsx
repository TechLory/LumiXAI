interface InputPanelProps {
  inputText: string;
  setInputText: (text: string) => void;
  onExplainClick: () => void;
  inferenceStatus?: 'idle' | 'running' | 'success' | 'error' | string; 
  isConfigReady: boolean;
}

export default function InputPanel(props: InputPanelProps) {
  const isRunning = props.inferenceStatus === 'running';
  const wordCount = props.inputText.trim().split(/\s+/).filter((word: string) => word.length > 0).length;

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

        <div className="text-neutral-500 font-mono text-xs text-center mt-1">
          Total words: {wordCount}
        </div>

        {/* RUN BUTTON */}
        <div className="mt-6">
          <button 
            className="bg-blue-900/40 hover:bg-blue-800/60 border border-blue-700/50 text-blue-400 w-full p-3 font-mono font-semibold text-sm uppercase cursor-pointer transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
            onClick={props.onExplainClick}
            disabled={isButtonDisabled} 
          >
            {!props.isConfigReady ? (
               <><i className='bx bx-lock-alt text-lg'></i> Waiting for Configuration...</>
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