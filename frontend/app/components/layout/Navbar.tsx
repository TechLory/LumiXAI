import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="w-full bg-yellow-600 text-black px-10 py-2 mb-2 flex justify-between font-mono font-semibold">
      <div className="select-none">LumiXAI</div>
      <div className="flex gap-10">
        <Link className="hover:underline underline-offset-4 decoration-2" href={""}>Docs</Link>
        <Link className="hover:underline underline-offset-4 decoration-2" href={""}>GitHub</Link>
      </div>
    </nav>
  );
}