"use client";

import { useState, useEffect } from "react";

interface StatusData {
  device: string;
  mixup_regressor: { loaded: boolean; model_path: string; exists: boolean };
  synthetic_regressor: { loaded: boolean; model_path: string; exists: boolean };
  translation_paths: {
    mixup: string;
    mixup_exists: boolean;
    synthetic: string;
    synthetic_exists: boolean;
    mixup_sft: string;
    mixup_sft_exists: boolean;
    synthetic_sft: string;
    synthetic_sft_exists: boolean;
  };
}

interface EvalResult {
  mixup_score: number;
  synthetic_score: number;
  stats: {
    token_count: number;
    sentence_count: number;
    avg_sentence_length: number;
  };
}

interface TranslateResult {
  translation: string;
  model_used: string;
  source_simplicity: { mixup: number; synthetic: number };
  target_simplicity: { mixup: number; synthetic: number };
}

export default function Home() {
  // State variables
  const [activeTab, setActiveTab] = useState<"evaluate" | "translate">("evaluate");
  const [status, setStatus] = useState<StatusData | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [evalText, setEvalText] = useState("");
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);
  const [loadingEval, setLoadingEval] = useState(false);

  const [translateText, setTranslateText] = useState("");
  const [selectedModel, setSelectedModel] = useState<"mixup" | "synthetic">("mixup");
  const [selectedTuning, setSelectedTuning] = useState<"dpo" | "sft">("dpo");
  const [translationResult, setTranslationResult] = useState<TranslateResult | null>(null);
  const [loadingTranslate, setLoadingTranslate] = useState(false);

  // Fetch status on mount
  useEffect(() => {
    fetchStatus();
  }, []);

  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [`[${timestamp}] ${msg}`, ...prev.slice(0, 19)]);
  };

  const fetchStatus = async () => {
    try {
      addLog("Lade Systemstatus vom Backend...");
      const res = await fetch("http://localhost:8000/api/status");
      if (res.ok) {
        const data: StatusData = await res.json();
        setStatus(data);
        addLog(`Systemstatus erfolgreich geladen. Device: ${data.device}`);
        
        if (data.mixup_regressor.loaded) {
          addLog("✅ MixUp Regressor (Bewertung) erfolgreich geladen.");
        } else {
          addLog("❌ MixUp Regressor (Bewertung) fehlt.");
        }
        if (data.synthetic_regressor.loaded) {
          addLog("✅ Synthetic Regressor (Bewertung) erfolgreich geladen.");
        } else {
          addLog("❌ Synthetic Regressor (Bewertung) fehlt.");
        }

        if (data.translation_paths.mixup_exists) {
          addLog("✅ MixUp-DPO Übersetzungsmodell gefunden.");
        } else {
          addLog(`⚠️ MixUp-DPO Übersetzungsmodell fehlt unter '${data.translation_paths.mixup}'`);
        }
        if (data.translation_paths.mixup_sft_exists) {
          addLog("✅ MixUp-SFT Übersetzungsmodell gefunden.");
        } else {
          addLog(`⚠️ MixUp-SFT Übersetzungsmodell (.pt) fehlt unter '${data.translation_paths.mixup_sft}'`);
        }
        if (data.translation_paths.synthetic_exists) {
          addLog("✅ Synthetic-DPO Übersetzungsmodell gefunden.");
        } else {
          addLog(`⚠️ Synthetic-DPO Übersetzungsmodell fehlt unter '${data.translation_paths.synthetic}'`);
        }
        if (data.translation_paths.synthetic_sft_exists) {
          addLog("✅ Synthetic-SFT Übersetzungsmodell gefunden.");
        } else {
          addLog(`⚠️ Synthetic-SFT Übersetzungsmodell (.pt) fehlt unter '${data.translation_paths.synthetic_sft}'`);
        }
      }
    } catch (err) {
      addLog("❌ Fehler beim Laden des Systemstatus. Ist das Backend aktiv?");
      console.error("Failed to fetch backend status:", err);
    }
  };

  const handleEvaluate = async () => {
    if (!evalText.trim()) return;
    setLoadingEval(true);
    addLog("Starte Bewertung...");
    try {
      const res = await fetch("http://localhost:8000/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: evalText }),
      });
      if (res.ok) {
        const data = await res.json();
        setEvalResult(data);
        addLog("✅ Bewertung abgeschlossen.");
      } else {
        addLog("❌ Fehler bei der Bewertung.");
      }
    } catch (err) {
      addLog("❌ Netzwerkfehler bei der Bewertung.");
      console.error("Error during evaluation:", err);
    } finally {
      setLoadingEval(false);
    }
  };

  const handleTranslate = async () => {
    if (!translateText.trim()) return;
    setLoadingTranslate(true);
    setTranslationResult(null);
    const modelName = `${selectedModel === "mixup" ? "MixUp" : "Synthetic"}-${selectedTuning.toUpperCase()}`;
    addLog(`Starte Übersetzung mit ${modelName} Modell...`);
    try {
      const res = await fetch("http://localhost:8000/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: translateText, model_type: selectedModel, tuning_type: selectedTuning }),
      });
      if (res.ok) {
        const data = await res.json();
        setTranslationResult(data);
        addLog("✅ Übersetzung erfolgreich abgeschlossen.");
      } else {
        const errData = await res.json();
        addLog(`❌ Fehler bei Übersetzung: ${errData.detail || "Unbekannter Fehler"}`);
      }
    } catch (err) {
      addLog("❌ Netzwerkfehler bei der Übersetzung.");
      console.error("Error during translation:", err);
    } finally {
      setLoadingTranslate(false);
    }
  };

  // Helper values for percentages
  const mixPct = evalResult ? Math.round(evalResult.mixup_score * 100) : 0;
  const synPct = evalResult ? Math.round(evalResult.synthetic_score * 100) : 0;

  const sourceSimp = translationResult ? Math.round(translationResult.source_simplicity[selectedModel] * 100) : 0;
  const targetSimp = translationResult ? Math.round(translationResult.target_simplicity[selectedModel] * 100) : 0;
  const delta = targetSimp - sourceSimp;

  const modelExists = selectedModel === "mixup" 
    ? (selectedTuning === "dpo" ? status?.translation_paths.mixup_exists : status?.translation_paths.mixup_sft_exists)
    : (selectedTuning === "dpo" ? status?.translation_paths.synthetic_exists : status?.translation_paths.synthetic_sft_exists);

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 flex items-center justify-between px-8 py-4 border-b border-slate-200 bg-white/95 backdrop-blur-md">
        <div>
          <h1 className="font-display text-lg font-bold tracking-tight text-slate-900 flex items-center gap-2">
            Masterarbeit <span className="logo-badge">Cockpit</span>
          </h1>
        </div>
        <div className="flex gap-4 text-xs text-slate-600">
          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-full">
            <span className={`status-dot ${status?.mixup_regressor.loaded ? "active" : "inactive"}`}></span>
            <span>MixUp Regressor</span>
          </div>
          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-full">
            <span className={`status-dot ${status?.synthetic_regressor.loaded ? "active" : "inactive"}`}></span>
            <span>Synthetic Regressor</span>
          </div>
          <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-full">
            <span className="font-semibold text-slate-800">Device:</span>
            <span className="uppercase text-[#6366f1] font-bold">{status?.device || "CPU"}</span>
          </div>
        </div>
      </header>

      {/* Tab Selector */}
      <div className="w-full max-w-[800px] mx-auto px-8 pt-8 flex justify-center">
        <div className="flex bg-slate-100/80 border border-slate-200/65 p-1.5 rounded-2xl w-full shadow-inner">
          <button
            onClick={() => setActiveTab("evaluate")}
            className={`flex-1 py-3 px-4 rounded-xl text-sm font-semibold font-display transition-all duration-200 text-center flex items-center justify-center gap-2 cursor-pointer ${
              activeTab === "evaluate"
                ? "bg-white text-[#6366f1] shadow-sm font-bold border border-slate-200/30"
                : "text-slate-555 hover:text-slate-800"
            }`}
          >
            Linguistische Bewertung
          </button>
          <button
            onClick={() => setActiveTab("translate")}
            className={`flex-1 py-3 px-4 rounded-xl text-sm font-semibold font-display transition-all duration-200 text-center flex items-center justify-center gap-2 cursor-pointer ${
              activeTab === "translate"
                ? "bg-white text-[#6366f1] shadow-sm font-bold border border-slate-200/30"
                : "text-slate-555 hover:text-slate-800"
            }`}
          >
            Übersetzung in Leichte Sprache
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-[800px] mx-auto px-8 py-8">
        {activeTab === "evaluate" ? (
          /* Rating Card */
          <div className="glass-card animate-in fade-in slide-in-from-bottom-2 duration-300">
            <h2 className="font-display text-base font-bold text-slate-800 border-b border-slate-100 pb-3 flex items-center gap-2">
              Linguistische Bewertung
            </h2>
            <div className="flex flex-col gap-2 relative">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Text zur Bewertung eingeben
              </label>
              <textarea
                value={evalText}
                onChange={(e) => setEvalText(e.target.value)}
                placeholder="Geben Sie hier Ihren alltagssprachlichen oder leichtsprachlichen Text ein..."
                className="w-full min-h-[180px] bg-slate-50/50 border border-slate-200 rounded-xl p-4 text-slate-850 text-sm leading-relaxed focus:outline-none focus:border-[#6366f1] focus:ring-3 focus:ring-[#6366f1]/10 transition-all resize-y"
              />
              <div className="absolute bottom-2 right-4 text-xs text-slate-400">
                {evalText.length} Zeichen
              </div>
            </div>
            <div>
              <button
                onClick={handleEvaluate}
                disabled={loadingEval || !evalText.trim()}
                className="px-5 py-2.5 bg-[#6366f1] hover:bg-[#4f46e5] disabled:bg-slate-200 disabled:cursor-not-allowed text-white font-semibold text-sm rounded-lg shadow-sm hover:shadow transition-all flex items-center gap-2"
              >
                {loadingEval ? (
                  <>
                    <div className="loading-spinner h-3.5 w-3.5 border-indigo-200 border-t-white" />
                    <span>Wird berechnet...</span>
                  </>
                ) : (
                  <span>Text Bewerten</span>
                )}
              </button>
            </div>

            {evalResult && (
              <div className="bg-slate-50/70 border border-slate-200 rounded-xl p-6 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                  Ergebnis der Einfachheits-Bewertung
                </h3>
                <div className="flex justify-around items-center gap-6 my-2">
                  <div className="flex flex-col items-center gap-3 bg-white border border-slate-100 p-5 rounded-xl w-[48%] text-center shadow-xs">
                    <div
                      className="circular-progress"
                      style={{
                        background: `conic-gradient(var(--primary) ${mixPct * 3.6}deg, #f1f5f9 0deg)`,
                      }}
                    >
                      <span className="progress-value">{mixPct}%</span>
                    </div>
                    <div className="text-xs font-bold text-slate-600">MixUp-Modell</div>
                  </div>
                  <div className="flex flex-col items-center gap-3 bg-white border border-slate-100 p-5 rounded-xl w-[48%] text-center shadow-xs">
                    <div
                      className="circular-progress"
                      style={{
                        background: `conic-gradient(var(--accent-teal) ${synPct * 3.6}deg, #f1f5f9 0deg)`,
                      }}
                    >
                      <span className="progress-value">{synPct}%</span>
                    </div>
                    <div className="text-xs font-bold text-slate-600">Synthetisches Modell</div>
                  </div>
                </div>
                <div className="flex justify-between border-t border-slate-100 pt-4 text-xs text-slate-500">
                  <span>Sätze: <strong className="text-slate-800">{evalResult.stats.sentence_count}</strong></span>
                  <span>Wörter: <strong className="text-slate-800">{evalResult.stats.token_count}</strong></span>
                  <span>Ø Satzlänge: <strong className="text-slate-800">{evalResult.stats.avg_sentence_length} Wörter</strong></span>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Translation Card */
          <div className="glass-card animate-in fade-in slide-in-from-bottom-2 duration-300">
            <h2 className="font-display text-base font-bold text-slate-800 border-b border-slate-100 pb-3 flex items-center gap-2">
              Übersetzung in Leichte Sprache
            </h2>
            <div className="flex flex-col gap-2">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Quelltext (Alltagssprache)
              </label>
              <textarea
                value={translateText}
                onChange={(e) => setTranslateText(e.target.value)}
                placeholder="Geben Sie hier den zu übersetzenden Text ein..."
                className="w-full min-h-[180px] bg-slate-50/50 border border-slate-200 rounded-xl p-4 text-slate-850 text-sm leading-relaxed focus:outline-none focus:border-[#6366f1] focus:ring-3 focus:ring-[#6366f1]/10 transition-all resize-y"
              />
            </div>
            
            <div className="flex flex-col gap-4 py-2 border-y border-slate-100 my-2">
              <div className="flex justify-between items-center gap-4 flex-wrap">
                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Ausrichtung:</span>
                  <div className="flex bg-slate-100 border border-slate-250 p-1 rounded-lg gap-1">
                    <button
                      onClick={() => setSelectedModel("mixup")}
                      className={`px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                        selectedModel === "mixup" ? "bg-[#6366f1] text-white shadow-xs" : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      MixUp Reward
                    </button>
                    <button
                      onClick={() => setSelectedModel("synthetic")}
                      className={`px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                        selectedModel === "synthetic" ? "bg-[#6366f1] text-white shadow-xs" : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      Synthetic Reward
                    </button>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Modell-Typ:</span>
                  <div className="flex bg-slate-100 border border-slate-250 p-1 rounded-lg gap-1">
                    <button
                      onClick={() => setSelectedTuning("dpo")}
                      className={`px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                        selectedTuning === "dpo" ? "bg-[#6366f1] text-white shadow-xs" : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      DPO
                    </button>
                    <button
                      onClick={() => setSelectedTuning("sft")}
                      className={`px-3 py-1 rounded-md text-xs font-semibold transition-all cursor-pointer ${
                        selectedTuning === "sft" ? "bg-[#6366f1] text-white shadow-xs" : "text-slate-600 hover:text-slate-900"
                      }`}
                    >
                      SFT
                    </button>
                  </div>
                </div>
              </div>

              <div className="flex justify-end">
                <button
                  onClick={handleTranslate}
                  disabled={loadingTranslate || !translateText.trim() || !modelExists}
                  className={`px-5 py-2.5 font-semibold text-sm rounded-lg shadow-sm hover:shadow transition-all flex items-center gap-2 cursor-pointer ${
                    modelExists 
                      ? "bg-[#6366f1] hover:bg-[#4f46e5] text-white" 
                      : "bg-slate-200 text-slate-400 cursor-not-allowed"
                  }`}
                >
                  {loadingTranslate ? (
                    <>
                      <div className="loading-spinner h-3.5 w-3.5 border-indigo-200 border-t-white" />
                      <span>Übersetzen...</span>
                    </>
                  ) : (
                    <span>Übersetzen</span>
                  )}
                </button>
              </div>
            </div>

            {/* Model Status Message */}
            {!modelExists && status && (
              <div className="bg-[#ef4444]/5 border border-[#ef4444]/20 p-4 rounded-xl text-xs text-[#ef4444] font-medium leading-relaxed">
                ⚠️ Übersetzung deaktiviert: Das ausgewählte Modell ({selectedModel === "mixup" ? "MixUp" : "Synthetic"}-{selectedTuning.toUpperCase()}) wurde auf dem System nicht gefunden.
                Legen Sie die Modellgewichte unter 
                <code className="bg-slate-100 border border-slate-200 px-1 py-0.5 rounded mx-1 text-slate-800">
                  {selectedModel === "mixup" 
                    ? (selectedTuning === "dpo" ? status.translation_paths.mixup : status.translation_paths.mixup_sft)
                    : (selectedTuning === "dpo" ? status.translation_paths.synthetic : status.translation_paths.synthetic_sft)}
                </code> 
                ab, um die Übersetzung zu aktivieren.
              </div>
            )}

            {/* Loader */}
            {loadingTranslate && (
              <div className="text-center py-8">
                <div className="loading-spinner h-6 w-6" />
                <p className="mt-4 text-sm text-[#475569]">Übersetzung wird generiert. Dies kann einen Moment dauern...</p>
              </div>
            )}

            {/* Translation Result Panel */}
            {translationResult && (
              <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-slate-50 border border-slate-100 p-4 rounded-lg text-sm leading-relaxed text-slate-600">
                    <div className="compare-badge badge-as">Alltagssprache</div>
                    <div>{translateText}</div>
                  </div>
                  <div className="bg-slate-50 border border-slate-100 p-4 rounded-lg text-sm leading-relaxed text-slate-800 font-medium">
                    <div className="compare-badge badge-ls">Leichte Sprache</div>
                    <div>{translationResult.translation}</div>
                  </div>
                </div>

                {/* Simplicity Delta Bar */}
                <div className="bg-slate-50/50 border border-slate-200 p-4 rounded-xl flex flex-col gap-2">
                  <div className="flex justify-between items-center text-xs font-semibold text-slate-600">
                    <span>Erhöhung der Einfachheit (Modell: <span className="text-[#6366f1] font-bold">{selectedModel === "mixup" ? "MixUp" : "Synthetic"}</span>)</span>
                    <span className="delta-badge bg-[#ecfdf5] text-[#10b981] font-bold text-xs px-2.5 py-1 rounded-full border border-[#a7f3d0]">
                      {delta >= 0 ? "+" : ""}{delta}%
                    </span>
                  </div>
                  <div className="h-3 w-full bg-slate-100 rounded-full overflow-hidden flex relative">
                    <div
                      className="h-full bg-sky-400 transition-all duration-500"
                      style={{ width: `${sourceSimp}%` }}
                    />
                    <div
                      className="h-full bg-[#10b981] transition-all duration-500"
                      style={{ width: `${Math.max(0, delta)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[11px] text-slate-500">
                    <span>Vorher: <strong className="text-slate-700">{sourceSimp}%</strong></span>
                    <span>Nachher: <strong className="text-slate-700">{targetSimp}%</strong></span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Logs Dashboard Widget */}
      <div className="max-w-[800px] w-full mx-auto px-8 mb-8">
        <div className="glass-card">
          <h3 className="font-display text-xs font-bold text-slate-800 border-b border-slate-100 pb-2 flex justify-between items-center">
            <span>System-Log / Modellstatus-Protokoll</span>
            <button 
              onClick={fetchStatus} 
              className="text-xs text-[#6366f1] hover:underline font-bold"
            >
              Status Aktualisieren
            </button>
          </h3>
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 font-mono text-[11px] text-slate-600 h-[150px] overflow-y-auto flex flex-col gap-1.5 leading-normal">
            {logs.length === 0 ? (
              <span className="italic text-slate-400">Keine Protokolleinträge vorhanden.</span>
            ) : (
              logs.map((log, idx) => (
                <div key={idx} className="whitespace-pre-wrap">
                  {log}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center py-8 text-xs text-slate-400 border-t border-slate-200 mt-auto bg-white">
        <p>Master Thesis © 2026. Automatische Übersetzung und semantische Bewertung.</p>
      </footer>
    </div>
  );
}
