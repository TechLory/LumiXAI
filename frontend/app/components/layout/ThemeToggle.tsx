"use client";

import { useState, useEffect } from "react";

/**
 * Toggles between the light and dark theme by flipping the `.dark` class on
 * <html> and persisting the choice to localStorage. The initial class is set
 * before paint by the inline script in layout.tsx, so this component only reads
 * the current state on mount to keep its icon in sync.
 */
export default function ThemeToggle() {
  const [isDark, setIsDark] = useState<boolean | null>(null);

  useEffect(() => {
    setIsDark(document.documentElement.classList.contains("dark"));
  }, []);

  const toggle = () => {
    const next = !document.documentElement.classList.contains("dark");
    document.documentElement.classList.toggle("dark", next);
    document.documentElement.style.colorScheme = next ? "dark" : "light";
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      /* ignore storage errors (e.g. private mode) */
    }
    setIsDark(next);
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle color theme"
      title="Toggle light / dark theme"
      className="flex items-center cursor-pointer hover:opacity-70 transition-opacity"
      suppressHydrationWarning
    >
      {/* Show the icon of the theme you would switch TO. Render the light icon
          until mounted so the markup is stable during hydration. */}
      <i className={`bx ${isDark ? "bx-sun" : "bx-moon"} text-xl`} suppressHydrationWarning></i>
    </button>
  );
}
