"use client";

import { useState } from "react";

import MainApp from "./components/MainApp";
import WelcomeScreen from "./components/WelcomeScreen";
import { isExamplesOnlyMode } from "./lib/examplesOnlyMode";
import type { TutorialKind } from "./types";

type AppScreen = "welcome" | "tool";

export default function Home() {
  const [screen, setScreen] = useState<AppScreen>("welcome");
  const [activeTutorial, setActiveTutorial] = useState<TutorialKind | null>(null);
  const examplesOnly = isExamplesOnlyMode;

  const openTool = () => {
    if (examplesOnly) {
      setActiveTutorial("text-classification");
      setScreen("tool");
      return;
    }

    setActiveTutorial(null);
    setScreen("tool");
  };

  const openTutorial = (tutorial: TutorialKind) => {
    setActiveTutorial(tutorial);
    setScreen("tool");
  };

  const shouldShowWelcome = screen === "welcome" || (examplesOnly && !activeTutorial);

  if (shouldShowWelcome) {
    return <WelcomeScreen examplesOnly={examplesOnly} onEnterTool={openTool} onSelectTutorial={openTutorial} />;
  }

  return (
    <MainApp
      activeTutorial={activeTutorial}
      onOpenWelcome={() => setScreen("welcome")}
      onSelectTutorial={openTutorial}
      onCloseTutorial={() => {
        setActiveTutorial(null);
        if (examplesOnly) setScreen("welcome");
      }}
    />
  );
}
