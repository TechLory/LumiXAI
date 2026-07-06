"use client";

import { useState } from "react";

import MainApp from "./components/MainApp";
import WelcomeScreen from "./components/WelcomeScreen";
import type { TutorialKind } from "./types";

type AppScreen = "welcome" | "tool";

export default function Home() {
  const [screen, setScreen] = useState<AppScreen>("welcome");
  const [activeTutorial, setActiveTutorial] = useState<TutorialKind | null>(null);

  const openTool = () => {
    setActiveTutorial(null);
    setScreen("tool");
  };

  const openTutorial = (tutorial: TutorialKind) => {
    setActiveTutorial(tutorial);
    setScreen("tool");
  };

  if (screen === "welcome") {
    return <WelcomeScreen onEnterTool={openTool} onSelectTutorial={openTutorial} />;
  }

  return (
    <MainApp
      activeTutorial={activeTutorial}
      onOpenWelcome={() => setScreen("welcome")}
      onSelectTutorial={openTutorial}
    />
  );
}
