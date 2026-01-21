
interface InputPanelProps {
  inputText: any;
  setInputText: any;
  onExplainClick: any;
}

export default function InputPanel(props: InputPanelProps) {
  return (
    <div className="mb-20 px-6">
      <div className="text-3xl font-semibold py-7 text-center">Input</div>

      {/* INPUT PANEL */}
      <textarea
        className="bg-neutral-400/20 w-full rounded-xl min-h-48 p-4 text-neutral-300 font-semibold font-mono"
        name="inputText"
        id="inputText"
        value={props.inputText}
        onChange={(e) => props.setInputText(e.target.value)}
      ></textarea>


      <div className="text-neutral-500 font-semibold text-sm mt-2 text-center">
        Total words: {props.inputText.trim().split(/\s+/).filter((word: string) => word.length > 0).length}
      </div>

      {/* ACTION BUTTON */}
      <div className="bg-neutral-400/20 hover:bg-neutral-400/30 w-1/2 m-auto mt-6 p-3 text-center font-semibold cursor-pointer rounded-xl" onClick={props.onExplainClick}>Run and Explain</div>







    </div>
  );
}