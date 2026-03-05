import { log } from "console";

// DEPRECATED !!!

export default function Tooltip({ iconName, tooltipText }: { iconName?: string; tooltipText?: string }) {

  let backgroundColor: string;
  let backgroundColorSemiTransparent: string;
  let textColor: string;
  let iconCode: string;

  switch (iconName) {
    case 'error':
      backgroundColor = 'bg-red-500';
      backgroundColorSemiTransparent = 'bg-red-500/20';
      textColor = 'text-red-500';
      iconCode = 'bxs-error';
      break;
    case 'stop':
      backgroundColor = 'bg-neutral-400';
      backgroundColorSemiTransparent = 'bg-neutral-400/20';
      textColor = 'text-neutral-500';
      iconCode = "bx-x";
      break;
    case 'success':
      backgroundColor = 'bg-green-400';
      backgroundColorSemiTransparent = 'bg-green-400/20';
      textColor = 'text-green-400';
      iconCode = 'bx-check';
      break;
    case 'success-unverified':
      backgroundColor = 'bg-yellow-400';
      backgroundColorSemiTransparent = 'bg-yellow-400/20';
      textColor = 'text-yellow-400';
      iconCode = 'bx-check';
      break;
    case 'loading':
      backgroundColor = 'bg-neutral-400';
      backgroundColorSemiTransparent = 'bg-neutral-400/20';
      textColor = 'text-neutral-300';
      iconCode = 'bx-loader-alt animate-spin';
      break;
    default:
      backgroundColor = 'bg-neutral-400';
      backgroundColorSemiTransparent = 'bg-neutral-400/20';
      textColor = 'text-neutral-500';
      iconCode = "bx-question-mark";
  }

 

  return (
    <div className="grow pr-6 text-right">

      <div className={`group relative inline-block cursor-pointer ${backgroundColorSemiTransparent} rounded-full`}> {/* BUG: color/20 non viene renderizzato - prova bg-opacity su div sfondo in stack */}
        <i className={`bx ${iconCode ?? "bx-question-mark"} text-6xl ${textColor} rounded-full p-5 transition`}></i>
        <div className={`invisible opacity-0 group-hover:visible group-hover:opacity-100 absolute bottom-full right-1/2 translate-x-1 mb-2 w-max p-5 ${backgroundColor} text-black font-mono rounded-xl font-semibold transition-opacity duration-300`}>
          {tooltipText ?? "Not available"}
        </div>
      </div>

    </div>
  );
}