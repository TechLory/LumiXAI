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
  setIsLoading: (loading: boolean) => void;
}


export default function ModelSelector(props: ModelSelectorProps) {
  const [query, setQuery] = useState(props.currentModel);
  const [results, setResults] = useState<HFModelResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  
  // use a Ref to track the debounce timer
  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  // use a Ref to detect clicks outside the component
  const wrapperRef = useRef<HTMLDivElement>(null);

  
  // handles text input changes and triggers the search with a debounce delay.
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    props.onModelSelect(val);

    if (val.length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    // Clear previous timer
    if (debounceRef.current) clearTimeout(debounceRef.current);

    // set new timer (500ms delay)
    debounceRef.current = setTimeout(() => {
      fetchModels(val);
    }, 500);
  };


  // fetches models from the backend API.
  const fetchModels = async (searchTerm: string) => {
    props.setIsLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/api/search?source=${encodeURIComponent(props.currentSource)}&q=${encodeURIComponent(searchTerm)}`);
      if (res.ok) {
        const data = await res.json();
        setResults(data);
        setIsOpen(true);
      }
    } catch (error) {
      console.error("Search failed", error);
    } finally {
      props.setIsLoading(false);
    }
  };


  // handles the selection of a model from the dropdown list.
  const selectModel = (model: HFModelResult) => {
    setQuery(model.id);
    props.onModelSelect(model.id);
    setIsOpen(false);
  };


  // closes dropdown when clicking outside
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
      
      <div className="relative">
        <input
          disabled={props.currentSource === ""}
          type="text"
          className="w-full p-2 border rounded mt-1 text-neutral-400"
          placeholder="Type to search..."
          value={query}
          onChange={handleInputChange}
          onFocus={() => results.length > 0 && setIsOpen(true)}
        />
        
      </div>

      {/* Dropdown Results */}
      {isOpen && results.length > 0 && (
        <ul className="absolute z-10 w-full bg-white border mt-1 rounded shadow-lg max-h-60 overflow-y-auto">
          {results.map((model) => (
            <li
              key={model.id}
              onClick={() => selectModel(model)}
              className="p-3 hover:bg-blue-50 cursor-pointer border-b last:border-b-0 transition-colors"
            >
              <div className="font-medium text-gray-900">{model.id}</div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span className="bg-gray-100 px-2 py-0.5 rounded text-gray-600">
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