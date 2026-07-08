"use client";
import { useState, useEffect, useRef } from "react";
import { buildApiUrl } from "../../lib/api";

interface HFModelResult {
  id: string;
  task: string;
  likes: number;
  downloads: number;
}

interface ModelSelectorProps {
  currentSource: string;
  currentModel: string;
  onModelSelect: (modelId: string, task?: string) => void;
  disabled?: boolean;
}

const MODEL_SEARCH_LIMIT = 25;

export default function ModelSelector(props: ModelSelectorProps) {
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
    // Task is unknown for a manually-typed model id (not picked from search results).
    props.onModelSelect(val, undefined);

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
      const url = new URL(buildApiUrl("/api/search"));
      url.searchParams.set("source", props.currentSource);
      url.searchParams.set("q", searchTerm);
      url.searchParams.set("limit", String(MODEL_SEARCH_LIMIT));
      const res = await fetch(url.toString());
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
    props.onModelSelect(model.id, model.task);
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
    <div className="relative w-full min-w-0" ref={wrapperRef}>
      
      <div className="relative flex min-w-0 items-center">
        <input
          disabled={props.currentSource === "" || props.disabled}
          type="text"
          className="w-full text-sm font-mono font-medium bg-transparent text-fg outline-none p-2 disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-fg-faint"
          placeholder="Type to search..."
          value={query}
          onChange={handleInputChange}
          onFocus={() => results.length > 0 && setIsOpen(true)}
        />
        {isSearching && (
          <i className='bx bx-loader animate-spin text-fg-faint absolute right-2'></i>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <ul className="absolute z-20 w-full bg-surface border border-border mt-1 shadow-xl max-h-96 overflow-y-auto font-mono text-sm">
          {results.map((model) => (
            <li
              key={model.id}
              onClick={() => selectModel(model)}
              className="p-3 hover:bg-fill cursor-pointer border-b border-border last:border-b-0 transition-colors flex flex-col gap-1"
            >
              <div className="font-medium text-fg truncate">{model.id}</div>
              <div className="flex flex-wrap justify-between gap-2 text-xs text-fg-subtle">
                <span className="bg-fill px-2 py-0.5 text-fg-muted">
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
