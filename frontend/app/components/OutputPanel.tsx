import TokenExplained from "./TokenExplained";

interface OutputPanelProps {
  outputResult: any;
}

export default function OutputPanel(props: OutputPanelProps) {
  return (
    <div className="mb-20 px-6">

      <div className="text-3xl font-semibold">Output</div>



      <div className="flex gap-1 mt-5">
        {props.outputResult.tokens?.map((token: string, index: number) => (
          <TokenExplained token={token} score={props.outputResult.scores?.[index]} />
        ))}
      </div>

      <div className="font-mono text-xl flex gap-2 mt-5">
        <div className="italic text-neutral-400">Model output: </div>
        {props.outputResult.target_id === 0 ? (
          <div className="font-bold">NEGATIVE</div>
        ) : props.outputResult.target_id === 1 ? (
          <div className="font-bold">POSITIVE</div>
        ) : (
          <div className="font-bold">{props.outputResult.predicted_token}</div>
        )}
      </div>

    </div>
  );
}