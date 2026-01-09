


export default function Tooltip({ iconName, tooltipText }: { iconName?: string; tooltipText?: string }) {

  let backgroundColor: string;
  let textColor: string;
  let iconCode: string;

  switch (iconName) {
    case 'error':
      backgroundColor = 'bg-red-500';
      textColor = 'text-red-500';
      iconCode = 'bxs-error';
      break;
    case 'success':
      backgroundColor = 'bg-green-400';
      textColor = 'text-green-400';
      iconCode = 'bx-check';
      break;
    case 'loading':
      backgroundColor = 'bg-neutral-400';
      textColor = 'text-neutral-300';
      iconCode = 'bx-loader-alt bx-spin';
      break;
    default:
      backgroundColor = 'bg-neutral-400';
      textColor = 'text-neutral-500';
      iconCode = "bx-question-mark";
  }


  return (
    <div className="grow pr-6 text-right">

      <div className={`group relative inline-block cursor-pointer ${backgroundColor}/20 rounded-full`}>
        <i className={`bx ${iconCode ?? "bx-question-mark"} text-6xl ${textColor} rounded-full p-5 transition`}></i>
        <div className={`invisible opacity-0 group-hover:visible group-hover:opacity-100 absolute bottom-full right-1/2 translate-x-1 mb-2 w-max p-5 ${backgroundColor} text-white rounded-xl font-semibold transition-opacity duration-300`}>
          {tooltipText ?? "Not available"}
        </div>
      </div>

    </div>
  );
}