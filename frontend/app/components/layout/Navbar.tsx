import Link from "next/link";
import ThemeToggle from "./ThemeToggle";

export default function Navbar() {
  return (
    <nav className="w-full bg-yellow-600 text-black px-10 py-2 mb-2 flex justify-between items-center font-mono font-semibold">
      <div className="select-none">LumiXAI</div>
      <div className="flex gap-10 items-center">
        <Link className="hover:underline underline-offset-4 decoration-2" href={"http://localhost:8001"} target="_blank">Docs</Link>
        <Link className="hover:underline underline-offset-4 decoration-2" href={"https://github.com/TechLory/xai-framework-lorenzo-gatta"} target="_blank">GitHub</Link>
        <ThemeToggle />
      </div>
    </nav>
  );
}