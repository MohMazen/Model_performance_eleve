import React, { useState, useEffect } from 'react';

declare global {
  interface Window {
    electronAPI: {
      selectFile: () => Promise<string | undefined>;
      runAnalysis: (filePath: string) => Promise<any>;
      onAnalysisLog: (callback: (message: string) => void) => void;
      onAnalysisError: (callback: (message: string) => void) => void;
    };
  }
}

function App() {
  const [filePath, setFilePath] = useState<string>('');
  const [logs, setLogs] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    if (window.electronAPI) {
        window.electronAPI.onAnalysisLog((message) => {
            setLogs((prev) => [...prev, message]);
        });
        window.electronAPI.onAnalysisError((message) => {
             setLogs((prev) => [...prev, `ERROR: ${message}`]);
        });
    }
  }, []);

  const handleSelectFile = async () => {
    if (window.electronAPI) {
      const path = await window.electronAPI.selectFile();
      if (path) {
        setFilePath(path);
      }
    }
  };

  const handleOpenQuestionnaire = () => {
    window.open('./Questionnaire.html', '_blank', 'width=1000,height=800');
  };

  const handleRunAnalysis = async () => {
    if (!filePath) return;
    setIsAnalyzing(true);
    setLogs([]);
    try {
      const res = await window.electronAPI.runAnalysis(filePath);
      setResult(res);
      setLogs((prev) => [...prev, 'Analysis complete!']);
    } catch (error: any) {
      setLogs((prev) => [...prev, `Failed: ${error}`]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Education Analysis Tool</h1>

      <div style={{ marginBottom: '20px' }}>
        <button onClick={handleOpenQuestionnaire} style={{ marginRight: '10px' }}>Open Questionnaire</button>
        <button onClick={handleSelectFile}>Select Excel File</button>
        <span style={{ marginLeft: '10px' }}>{filePath || 'No file selected'}</span>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <button
            onClick={handleRunAnalysis}
            disabled={!filePath || isAnalyzing}
            style={{
                padding: '10px 20px',
                fontSize: '16px',
                backgroundColor: isAnalyzing ? '#ccc' : '#007bff',
                color: 'white',
                border: 'none',
                cursor: isAnalyzing ? 'not-allowed' : 'pointer'
            }}
        >
          {isAnalyzing ? 'Running Analysis...' : 'Run Analysis'}
        </button>
      </div>

      <div style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1, border: '1px solid #ccc', padding: '10px', height: '400px', overflowY: 'auto', backgroundColor: '#f9f9f9' }}>
          <h3>Logs</h3>
          {logs.map((log, i) => (
            <div key={i} style={{ whiteSpace: 'pre-wrap', marginBottom: '5px' }}>{log}</div>
          ))}
        </div>
      </div>

      {result && (
        <div style={{ marginTop: '20px' }}>
            <h3>Results</h3>
            <p>Output saved to: {result.outputDir}</p>
            {/* Here we could display generated images if we update the python script to return paths */}
        </div>
      )}
    </div>
  );
}

export default App;