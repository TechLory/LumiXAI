"use client";

import Image from "next/image";
import Link from "next/link";
import { useRef } from "react";

import { useDocsUrl } from "../../lib/api";
import type { TutorialKind } from "../../types";
import ThemeToggle from "./ThemeToggle";

type NavbarProps = {
  activeTutorial?: TutorialKind | null;
  onOpenWelcome?: () => void;
  onSelectTutorial?: (tutorial: TutorialKind) => void;
};

type TutorialNavItem = {
  kind: TutorialKind;
  label: string;
  icon: string;
};

const tutorialItems: TutorialNavItem[] = [
  { kind: "text-classification", label: "Text classification", icon: "bx-category-alt" },
  { kind: "text-generation", label: "Text generation", icon: "bx-text" },
  { kind: "txt2img-generation", label: "Text to image", icon: "bx-image-add" },
  { kind: "image-classification", label: "Image classification", icon: "bx-grid-alt" },
];

function LogoMark() {
  return (
    <>
      <Image
        src="/logo-lightmode.svg"
        alt=""
        width={191}
        height={39}
        priority
        className="h-8 w-auto max-w-[46vw] dark:hidden"
      />
      <Image
        src="/logo-darkmode.svg"
        alt=""
        width={191}
        height={39}
        priority
        className="hidden h-8 w-auto max-w-[46vw] dark:block"
      />
    </>
  );
}

export default function Navbar({ activeTutorial = null, onOpenWelcome, onSelectTutorial }: NavbarProps) {
  const docsUrl = useDocsUrl();
  const tutorialsRef = useRef<HTMLDetailsElement>(null);

  const handleSelectTutorial = (tutorial: TutorialKind) => {
    onSelectTutorial?.(tutorial);
    if (tutorialsRef.current) {
      tutorialsRef.current.open = false;
    }
  };

  return (
    <nav className="w-full bg-surface text-fg border-b border-border px-4 sm:px-10 py-3 mb-2 flex flex-wrap justify-between items-center gap-3 font-mono font-semibold">
      {onOpenWelcome ? (
        <button
          type="button"
          onClick={onOpenWelcome}
          aria-label="LumiXAI welcome screen"
          className="inline-flex shrink-0 items-center cursor-pointer focus:outline-none focus:ring-2 focus:ring-info focus:ring-offset-2 focus:ring-offset-surface"
        >
          <LogoMark />
        </button>
      ) : (
        <Link href="/" aria-label="LumiXAI home" className="inline-flex shrink-0 items-center">
          <LogoMark />
        </Link>
      )}

      <div className="flex flex-wrap justify-end gap-x-4 gap-y-2 sm:gap-x-10 items-center text-sm sm:text-base">
        {onSelectTutorial && (
          <details ref={tutorialsRef} className="navbar-tutorials relative">
            <summary className="flex cursor-pointer items-center gap-1 hover:underline underline-offset-4 decoration-2">
              Tutorials
              <i className="bx bx-chevron-down text-lg" aria-hidden="true"></i>
            </summary>
            <div className="absolute right-0 top-full z-50 mt-3 w-64 border-2 border-border-strong bg-surface p-2 shadow-[4px_4px_0_var(--border-strong)]">
              {tutorialItems.map((item) => {
                const isActive = item.kind === activeTutorial;

                return (
                  <button
                    key={item.kind}
                    type="button"
                    onClick={() => handleSelectTutorial(item.kind)}
                    className={`flex w-full items-center gap-3 px-3 py-2 text-left text-sm uppercase transition-colors hover:bg-fill ${isActive ? "bg-info-soft text-info" : "text-fg-muted"}`}
                  >
                    <i className={`bx ${item.icon} text-lg`} aria-hidden="true"></i>
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </div>
          </details>
        )}
        <Link className="hover:underline underline-offset-4 decoration-2" href={docsUrl} target="_blank" rel="noreferrer">Docs</Link>
        <Link className="hover:underline underline-offset-4 decoration-2" href={"https://github.com/TechLory/xai-framework-lorenzo-gatta"} target="_blank" rel="noreferrer">GitHub</Link>
        <ThemeToggle />
      </div>
    </nav>
  );
}
