import Image from "next/image";


export default function Navbar() {
  return (
    <nav className="w-full bg-neutral-700 text-neutral-300 py-4 mb-10 flex justify-between items-center px-5">
      <div className="flex-1">
        <Image className="w-20" src="/unimi-logo.png" alt="XAI Framework Logo" width={200} height={200} />
      </div>
      <div className="flex-1 text-center font-mono font-semibold">XAI Framework</div>
      <div className="flex-1 text-right">
        <i className='bx bxl-github text-4xl'></i>
      </div>
    </nav>
  );
}