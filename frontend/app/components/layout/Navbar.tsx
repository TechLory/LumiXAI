import Image from "next/image";
import Link from "next/link";
import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  return (
    <nav className="w-full bg-surface text-fg border-b border-border px-4 sm:px-10 py-3 mb-2 flex justify-between items-center font-mono font-semibold">
      <Link href="/" aria-label="LumiXAI home" className="inline-flex shrink-0 items-center">
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
      </Link>
      <div className="flex gap-4 sm:gap-10 items-center text-sm sm:text-base">
        <Link className="hover:underline underline-offset-4 decoration-2" href={"http://localhost:8001"} target="_blank" rel="noreferrer">Docs</Link>
        <Link className="hover:underline underline-offset-4 decoration-2" href={"https://github.com/TechLory/xai-framework-lorenzo-gatta"} target="_blank" rel="noreferrer">GitHub</Link>
        <ThemeToggle />
      </div>
    </nav>
  );
}
