import type { TutorialKind } from "../../types";
import type { TutorialStep } from "../../lib/tutorialGuide";

type TutorialOverlayProps = {
  tutorialKind: TutorialKind;
  step: TutorialStep;
  stepIndex: number;
  stepCount: number;
  onBack: () => void;
  onNext: () => void;
  onClose: () => void;
};

const tutorialLabels: Record<TutorialKind, string> = {
  "text-classification": "Text classification",
  "text-generation": "Text generation",
  "txt2img-generation": "Text to image",
};

export default function TutorialOverlay({
  tutorialKind,
  step,
  stepIndex,
  stepCount,
  onBack,
  onNext,
  onClose,
}: TutorialOverlayProps) {
  const isLastStep = stepIndex === stepCount - 1;

  return (
    <>
      <div className="tutorial-scrim" aria-hidden="true" />
      <section
        className="tutorial-card fixed bottom-5 right-5 z-[70] w-[min(430px,calc(100vw-2.5rem))] border-2 border-info-line bg-surface p-4 font-mono text-fg shadow-[8px_8px_0_var(--border-strong)]"
        aria-live="polite"
      >
        <div className="mb-3 flex items-start justify-between gap-4">
          <div>
            <div className="text-[11px] font-bold uppercase text-info">
              {tutorialLabels[tutorialKind]} tutorial // Step {stepIndex + 1} of {stepCount}
            </div>
            <h2 className="mt-1 text-base font-bold uppercase">{step.title}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 cursor-pointer text-fg-subtle hover:text-fg"
            aria-label="Close tutorial"
          >
            <i className="bx bx-x text-2xl" aria-hidden="true"></i>
          </button>
        </div>

        <p className="text-sm leading-6 text-fg-muted">{step.body}</p>

        <div className="mt-4 h-1.5 bg-fill" aria-hidden="true">
          <div
            className="h-full bg-info transition-all"
            style={{ width: `${((stepIndex + 1) / stepCount) * 100}%` }}
          />
        </div>

        <div className="mt-4 flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={onBack}
            disabled={stepIndex === 0}
            className="flex items-center gap-2 border border-border-strong bg-fill px-3 py-2 text-xs font-bold uppercase text-fg-muted transition-colors hover:bg-fill-strong disabled:cursor-not-allowed disabled:opacity-40"
          >
            <i className="bx bx-left-arrow-alt text-lg" aria-hidden="true"></i>
            Back
          </button>
          <button
            type="button"
            onClick={onNext}
            className="flex items-center gap-2 border border-info-line bg-info-soft px-4 py-2 text-xs font-bold uppercase text-info transition-colors hover:bg-info-hover"
          >
            {isLastStep ? "Finish" : "Next"}
            <i className={`bx ${isLastStep ? "bx-check" : "bx-right-arrow-alt"} text-lg`} aria-hidden="true"></i>
          </button>
        </div>
      </section>
    </>
  );
}
