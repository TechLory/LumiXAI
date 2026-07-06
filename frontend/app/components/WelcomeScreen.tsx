"use client";

import Image from "next/image";
import Link from "next/link";
import type { CSSProperties } from "react";

import type { TutorialKind } from "../types";
import ThemeToggle from "./layout/ThemeToggle";

type WelcomeScreenProps = {
  onEnterTool: () => void;
  onSelectTutorial: (tutorial: TutorialKind) => void;
};

type TutorialAction = {
  kind: TutorialKind;
  label: string;
  description: string;
  icon: string;
};

const tutorialActions: TutorialAction[] = [
  {
    kind: "text-classification",
    label: "Text classification",
    description: "See which words drive a real sentiment prediction",
    icon: "bx-category-alt",
  },
  {
    kind: "text-generation",
    label: "Text generation",
    description: "Trace a real reply back to the prompt that shaped it",
    icon: "bx-text",
  },
  {
    kind: "txt2img-generation",
    label: "Text to image",
    description: "Map prompt words onto the pixels they generated",
    icon: "bx-image-add",
  },
];

const heatCells = Array.from({ length: 96 }, (_, index) => index);
const sampleTokens = ["prompt", "model", "attr", "input", "output", "history"];

export default function WelcomeScreen({ onEnterTool, onSelectTutorial }: WelcomeScreenProps) {
  return (
    <main className="welcome-screen relative min-h-screen overflow-hidden bg-page text-fg font-mono">
      <div className="welcome-pixel-bg" aria-hidden="true" />

      <header className="relative z-10 flex flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-10">
        <div className="inline-flex items-center" aria-label="LumiXAI">
          <Image
            src="/logo-lightmode.svg"
            alt="LumiXAI"
            width={191}
            height={39}
            priority
            className="h-8 w-auto max-w-[46vw] dark:hidden"
          />
          <Image
            src="/logo-darkmode.svg"
            alt="LumiXAI"
            width={191}
            height={39}
            priority
            className="hidden h-8 w-auto max-w-[46vw] dark:block"
          />
        </div>
        <div className="flex items-center gap-4 text-sm font-semibold sm:gap-8">
          <Link className="hover:underline underline-offset-4 decoration-2" href="http://localhost:8001" target="_blank" rel="noreferrer">Docs</Link>
          <Link className="hover:underline underline-offset-4 decoration-2" href="https://github.com/TechLory/xai-framework-lorenzo-gatta" target="_blank" rel="noreferrer">GitHub</Link>
          <ThemeToggle />
        </div>
      </header>

      <section className="relative z-10 mx-auto grid min-h-[calc(100vh-76px)] w-full max-w-7xl items-center gap-8 px-5 pb-10 pt-4 sm:px-10 lg:grid-cols-[minmax(0,1fr)_minmax(340px,0.78fr)]">
        <div className="max-w-3xl">
          <div className="welcome-heat-text mb-5 inline-flex border-2 border-border-strong bg-surface/85 px-3 py-2 text-xs font-bold uppercase shadow-[4px_4px_0_var(--border-strong)]">
            Interactive XAI workstation
          </div>

          <h1 className="sr-only">LumiXAI</h1>
          <div className="mb-7 max-w-[min(650px,90vw)]">
            <Image
              src="/logo-lightmode.svg"
              alt=""
              width={720}
              height={147}
              priority
              className="h-auto w-full dark:hidden"
            />
            <Image
              src="/logo-darkmode.svg"
              alt=""
              width={720}
              height={147}
              priority
              className="hidden h-auto w-full dark:block"
            />
          </div>

          <p className="max-w-2xl text-base leading-7 text-fg-muted sm:text-lg">
            A modular explainability workspace for inspecting classifiers, text generators, and text-to-image models through interactive bidirectional heatmaps.
          </p>

          <div className="mt-8 grid gap-3 sm:grid-cols-2">
            <button
              type="button"
              onClick={onEnterTool}
              className="welcome-action welcome-action-primary group flex min-h-24 items-center gap-4 border-2 border-fg bg-fg px-4 py-3 text-left text-page transition-transform hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-info focus:ring-offset-2 focus:ring-offset-page"
            >
              <span className="flex h-11 w-11 shrink-0 items-center justify-center border-2 border-page bg-page text-fg">
                <i className="bx bx-right-arrow-alt text-2xl" aria-hidden="true"></i>
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-bold uppercase">Open tool</span>
                <span className="mt-1 block text-xs leading-5 opacity-80">Go directly to the attribution interface</span>
              </span>
            </button>

            {tutorialActions.map((action) => (
              <button
                key={action.kind}
                type="button"
                onClick={() => onSelectTutorial(action.kind)}
                className="welcome-action group flex min-h-24 items-center gap-4 border-2 border-border-strong bg-surface/95 px-4 py-3 text-left transition-transform hover:-translate-y-0.5 hover:border-fg focus:outline-none focus:ring-2 focus:ring-info focus:ring-offset-2 focus:ring-offset-page"
              >
                <span className="welcome-action-icon flex h-11 w-11 shrink-0 items-center justify-center border-2 border-border-strong bg-fill text-fg">
                  <i className={`bx ${action.icon} text-2xl`} aria-hidden="true"></i>
                </span>
                <span className="min-w-0">
                  <span className="block text-sm font-bold uppercase">{action.label}</span>
                  <span className="mt-1 block text-xs leading-5 text-fg-subtle">{action.description}</span>
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="welcome-preview relative min-h-[420px] overflow-hidden border-2 border-border-strong bg-surface/80 p-4 shadow-[8px_8px_0_var(--border-strong)]" aria-hidden="true">
          <div className="mb-4 flex items-center justify-between border-b-2 border-border-strong pb-3 text-xs uppercase text-fg-subtle">
            <span>{"// Attribution buffer"}</span>
            <span className="welcome-heat-text font-bold">live</span>
          </div>

          <div className="welcome-demo-grid grid grid-cols-12 gap-1">
            {heatCells.map((cell) => (
              <span
                key={cell}
                className="welcome-demo-cell block aspect-square"
                style={{
                  "--delay": `${-(cell % 19) * 0.12}s`,
                  "--heat-hue": `${(cell * 29) % 360}deg`,
                } as CSSProperties}
              />
            ))}
          </div>

          <div className="mt-6 grid grid-cols-2 gap-2">
            {sampleTokens.map((token, index) => (
              <div
                key={token}
                className="welcome-token flex items-center justify-between border-2 border-border bg-fill px-3 py-2 text-xs uppercase"
                style={{
                  "--delay": `${-index * 0.3}s`,
                  "--heat-hue": `${index * 51}deg`,
                } as CSSProperties}
              >
                <span>{token}</span>
                <span className="font-bold">{(0.91 - index * 0.09).toFixed(2)}</span>
              </div>
            ))}
          </div>

          <div className="welcome-scanline pointer-events-none absolute inset-0" />
        </div>
      </section>
    </main>
  );
}
