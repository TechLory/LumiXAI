"use client";

import { useEffect, useState } from "react";

import Navbar from "./components/Navbar";
import Header from "./components/Header";
import ConfigurationPanel from "./components/ConfigurationPanel";;
import InputPanel from "./components/InputPanel";
import OutputPanel from "./components/OutputPanel";


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
  // model loading
  LOADING_MODEL = "Loading model...",
  ERROR_LOADING_MODEL = "Error loading model:",
  MODEL_LOADED = "Model loaded successfully.\n",
  // generic error
  ERROR = "Something went wrong. Please ensure the backend server is running and try again.",
}

type Manifest = {
  sources: { id: string; name: string; type: string }[];
  attributors: { id: string; name: string }[];
};

export default function Home() {

  // system status
  const [isAppLoading, setIsAppLoading] = useState(true);
  const [isModelLoading, setIsModelLoading] = useState(false);

  const [isInferenceRunning, setIsInferenceRunning] = useState(false);
  const [isInferenceFailed, setIsInferenceFailed] = useState(false);
  const [inferenceError, setInferenceError] = useState("");

  const [statusLoadModel, setStatusLoadModel] = useState("No model loaded.");
  const [statusAttributor, setStatusAttributor] = useState("");
  const [isModelLoadedSuccessfully, setIsModelLoadedSuccessfully] = useState(false);
  const [error, setError] = useState("");


  // pre-checks
  const [preChecks, setPreChecks] = useState<Status[]>([]);

  // objects
  const [manifest, setManifest] = useState<Manifest | null>(null);

  // user selections
  const [selectedSource, setSelectedSource] = useState("");
  const [modelName, setModelName] = useState("");
  const [selectedAttributor, setSelectedAttributor] = useState("");

  const [inputText, setInputText] = useState("Astronauts riding horses on Mars.");
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
        //if (data.sources.length > 0) setSelectedSource(data.sources[0].id); // auto-select first source

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

    setIsModelLoading(true);
    setStatusLoadModel(Status.LOADING_MODEL);

    try {
      const res = await fetch("http://localhost:8000/api/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: selectedSource,
          model_name: modelName,
          device: "cuda" // TO FIX: put AUTO and choose cuda if available, fallback to cpu
        })
      });

      if (!res.ok) throw new Error(await res.text());
      setIsModelLoadedSuccessfully(true);
      const data = await res.json();
      console.log(data);
      const details = `Status: ${data.status} (${data.model})\nDetected model type: ${data.detected_task} (assigned to wrapper "${data.wrapper}")`;
      setStatusLoadModel(Status.MODEL_LOADED + details);

    } catch (e: any) {
      setStatusLoadModel(Status.ERROR_LOADING_MODEL + " " + JSON.parse(e.message).detail || e.message);
      setIsModelLoadedSuccessfully(false);
    }

    setIsModelLoading(false);
  };

  // action SetAttributor: set attributor in backend
  const handleSetAttributor = async (newValue: string) => {

    if (isModelLoadedSuccessfully) {

      setSelectedAttributor(newValue);

      try {
        const res = await fetch("http://localhost:8000/api/set_attributor", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ attributor_id: newValue })
        });

        if (!res.ok) throw new Error("Failed to set attributor");
        setStatusAttributor(`Attributor set successfully (${newValue})`);

      } catch (e: any) {
        setStatusAttributor("Error Attributor: " + e.message);
      }
    }
  };

  // action Explain: get explanation from backend
  const handleExplain = async () => {

    setIsInferenceRunning(true);
    setIsInferenceFailed(false);
    setInferenceError("");
    setOutputResult(null);

    console.log("calling /api/explain");

    try {
      const res = await fetch("http://localhost:8000/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText })
      });

      const data = await res.json();
      console.log("response from /api/explain:", data);

      if (!res.ok) throw new Error(data.detail || "Inference failed");

      setIsInferenceFailed(false);
      setOutputResult(data);
    } catch (e: any) {
      setInferenceError(e.message);
      setIsInferenceFailed(true);
    }

    setIsInferenceRunning(false);
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



              <ConfigurationPanel
                manifest={manifest}
                selectedSource={selectedSource}
                modelName={modelName}
                isModelLoading={isModelLoading}
                isModelLoadedSuccessfully={isModelLoadedSuccessfully}
                statusLoadModel={statusLoadModel}
                statusAttributor={statusAttributor}
                selectedAttributor={selectedAttributor}
                onSourceChange={setSelectedSource}
                onModelNameChange={setModelName}
                setIsModelLoadedSuccessfully={setIsModelLoadedSuccessfully}
                onLoadModelClick={handleLoadModel}
                onAttributorChange={handleSetAttributor}
              />

              <InputPanel
                inputText={inputText}
                setInputText={setInputText}
                onExplainClick={handleExplain}
              />

              {isInferenceRunning ? (
                <div className="w-full h-80 text-center">
                  <i className='bx bx-loader animate-spin text-5xl text-neutral-500'></i>
                </div>

              ) : isInferenceFailed ? (
                <div className="text-red-500">
                  <div>An error occurred during inference:</div>
                  <div>{inferenceError}</div>
                </div>

              ) : (
                <div>
                  {outputResult && (
                    <OutputPanel outputResult={outputResult} />
                  )}
                </div>
              )}


              <div className="h-60"></div>
            </div>
          </>
        )}
      </div>
    </div >
  );
}
