"use client";

import { useEffect, useState } from "react";

import Navbar from "./components/Navbar";
import Header from "./components/Header";
import ConfigurationPanel from "./components/ConfigurationPanel";
import InferencePanel from "./components/InferencePanel";
import VisualizationResult from "./components/VisualizationResult";
import ModelSelector from "./components/ModelSelector";

enum Status {
  // booting
  SYSTEM_BOOTING = "System booting...",
  // server connection
  CHECK_SERVER_CONNECTION = "Checking connection to server...",
  ERROR_SERVER_CONNECTION = "Error connecting to server. Please ensure the backend server is running.\nRun 'poetry run uvicorn main:app --reload' to start the server.",
  CONNECTION_ESTABLISHED = "Connection to server established.",
  // manifest loading
  LOADING_MANIFEST = "Loading manifest...",
  ERROR_LOADING_MANIFEST = "Error loading manifest.",
  MANIFEST_LOADED = "Manifest loaded.",
  // generic error
  ERROR = "Something went wrong. Please ensure the backend server is running and try again.",
}


export default function Home() {

  // system status
  const [isAppLoading, setIsAppLoading] = useState(true);
  const [error, setError] = useState("");

  // pre-checks
  const [preChecks, setPreChecks] = useState<Status[]>([]);

  // objects
  const [manifest, setManifest] = useState<{ wrappers: { id: string; name: string }[]; attributors: { id: string; name: string }[] } | null>(null); // TOFIX: da aggiornare attributor con id e name in main.py

  // user selections
  const [selectedWrapper, setSelectedWrapper] = useState("");
  const [modelName, setModelName] = useState("");
  const [selectedAttributor, setSelectedAttributor] = useState("");

  const [inputText, setInputText] = useState("Nel mezzo del cammin di nostra vita");
  const [outputResult, setOutputResult] = useState<any>(null);




  // initial loading
  useEffect(() => {
    const initSystem = async () => {
      try {
        // 1. check server
        setPreChecks(prev => [...prev, Status.CHECK_SERVER_CONNECTION]);
        const server = await fetch("http://localhost:8000");
        if (!server.ok) throw new Error(Status.ERROR_SERVER_CONNECTION);

        // DEBUG:
        //await new Promise(resolve => setTimeout(resolve, 1000)); // simulate delay
        //throw new Error(Status.ERROR_SERVER_CONNECTION); // simulate error

        // 2. load manifest
        setPreChecks(prev => [...prev, Status.LOADING_MANIFEST]);
        const res = await fetch("http://localhost:8000/api/manifest");
        const data = await res.json();
        setManifest(data);

        // all pre-checks passed
        setIsAppLoading(false);
      } catch (err: any) {
        setError(err.message || Status.ERROR);
        setIsAppLoading(false);
      }
    };

    setPreChecks(prev => [...prev, Status.SYSTEM_BOOTING]);
    initSystem();
  }, []);

  // action LoadModel: load model from backend
  const handleLoadModel = async () => {

    try {
      const res = await fetch("http://localhost:8000/api/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          wrapper_id: selectedWrapper,
          model_name: modelName,
          device: "cpu" // hardcoded (to be improved)
        })
      });
      if (!res.ok) throw new Error(await res.text());


      // auto-set attributor to speed up process
      await handleSetAttributor();
    } catch (e: any) {
      setError("Error Loading Model: " + e.message);
    }
  };

  // action Explain: get explanation from backend
  const handleExplain = async () => {

    try {
      const res = await fetch("http://localhost:8000/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText })
      });
      const data = await res.json();
      setOutputResult(data);
    } catch (e: any) {
      setError("Error Explaining: " + e.message);
    }
  };

  // action SetAttributor: set attributor in backend
  const handleSetAttributor = async () => {
    try {
      await fetch(`http://localhost:8000/api/set_attributor?attributor_id=${selectedAttributor}`, {
        method: "POST"
      });
    } catch (e: any) { setError("Error Attributor: " + e.message); }
  };











  return (
    <div className="bg-neutral-900 text-white min-h-screen">


      <div className="w-full md:w-5/6 lg:w-3/4 xl:w-2/3 2xl:w-1/2 m-auto bg-neutral-800 min-h-screen relative">
        <Navbar />
        <Header />

        <div className="flex flex-col px-5 mt-2">
          {preChecks.map((check, index) => (
            <div key={index} className="font-mono text-neutral-500 text-sm">{check}</div>
          ))}
        </div>

        {error ? (
          <div className="text-red-500 px-5 mt-10 text-lg font-semibold whitespace-pre-line">{error}</div>
        ) : isAppLoading ? (
          <div className="absolute top-0 w-full h-screen flex flex-col justify-center items-center">
            <i className='bx bx-loader animate-spin text-5xl text-neutral-500'></i>
          </div>
        ) : (
          <>
            <div className="px-5 font-mono text-green-500 text-sm">System ready.</div>
            <div className="mt-20">

              <div className="px-5 font-mono text-sm text-neutral-400">
                <pre>{JSON.stringify(manifest, null, 2)}</pre>
              </div>



              <ConfigurationPanel
                manifest={manifest}
                selectedWrapper={selectedWrapper}
                modelName={modelName}
                selectedAttributor={selectedAttributor}
                onWrapperChange={setSelectedWrapper}
                onModelNameChange={setModelName}
                onLoadClick={handleLoadModel}
                onAttributorChange={setSelectedAttributor}
              />

              <InferencePanel inputText={inputText} />

              <VisualizationResult result={outputResult} />






            </div>
          </>
        )}
      </div>
    </div >
  );
}
