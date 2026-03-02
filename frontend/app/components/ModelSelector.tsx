"use client";
import { useState, useEffect, useRef } from "react";

interface HFModelResult {
  id: string;
  task: string;
  likes: number;
  downloads: number;
}

interface ModelSelectorProps {
  currentSource: string;
  currentModel: string;
  onModelSelect: (modelId: string) => void;
}

export default function ModelSelector(props: ModelSelectorProps) {
  const ipAddress = "192.168.1.23";
  const [query, setQuery] = useState(props.currentModel);
  const [results, setResults] = useState<HFModelResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setQuery(props.currentModel);
  }, [props.currentModel]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    props.onModelSelect(val);

    if (val.length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    if (debounceRef.current) clearTimeout(debounceRef.current);

    debounceRef.current = setTimeout(() => {
      fetchModels(val);
    }, 500);
  };

  const fetchModels = async (searchTerm: string) => {
    setIsSearching(true);
    try {
      const res = await fetch(`http://${ipAddress}:8000/api/search?source=${encodeURIComponent(props.currentSource)}&q=${encodeURIComponent(searchTerm)}`);
      if (res.ok) {
        const data = await res.json();
        setResults(data);
        setIsOpen(true);
      }
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      setIsSearching(false);
    }
  };

  const selectModel = (model: HFModelResult) => {
    setQuery(model.id);
    props.onModelSelect(model.id);
    setIsOpen(false);
  };

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="relative w-full" ref={wrapperRef}>
      
      <div className="relative flex items-center">
        <input
          disabled={props.currentSource === ""}
          type="text"
          className="w-full text-sm font-mono font-medium bg-transparent text-white outline-none p-2 disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-neutral-300"
          placeholder="Type to search..."
          value={query}
          onChange={handleInputChange}
          onFocus={() => results.length > 0 && setIsOpen(true)}
        />
        {isSearching && (
          <i className='bx bx-loader animate-spin text-neutral-500 absolute right-2'></i>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <ul className="absolute z-20 w-full bg-neutral-800 border border-neutral-700 mt-1 shadow-xl max-h-60 overflow-y-auto font-mono text-sm">
          {results.map((model) => (
            <li
              key={model.id}
              onClick={() => selectModel(model)}
              className="p-3 hover:bg-neutral-700 cursor-pointer border-b border-neutral-700/50 last:border-b-0 transition-colors flex flex-col gap-1"
            >
              <div className="font-medium text-white truncate">{model.id}</div>
              <div className="flex justify-between text-xs text-neutral-400">
                <span className="bg-neutral-900 px-2 py-0.5 text-neutral-300">
                  {model.task}
                </span>
                <span>⬇ {model.downloads.toLocaleString()}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}