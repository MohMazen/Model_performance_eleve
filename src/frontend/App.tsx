import React, { useState, useEffect } from 'react';

// Define the interface for the exposed electron API
interface ElectronAPI {
  runAnalysis: (args?: any) => Promise<any>;
  onLog: (callback: (log: string) => void) => void;
  onError: (callback: (err: string) => void) => void;
}

declare global {
  interface Window {
    electron: ElectronAPI;
  }
}

const App: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>('Ready');
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    if (window.electron) {
        window.electron.onLog((log) => {
            setLogs((prev) => [...prev, log]);
        });
        window.electron.onError((err) => {
             setLogs((prev) => [...prev, `ERROR: ${err}`]);
        });
    }
  }, []);

  const handleRunAnalysis = async () => {
    if (!window.electron) {
        alert("Electron API not available. Are you running in browser?");
        return;
    }

    setIsRunning(true);
    setStatus('Running analysis...');
    setLogs([]); // Clear logs

    try {
      await window.electron.runAnalysis({});
      setStatus('Analysis Complete!');
    } catch (error) {
      console.error(error);
      setStatus('Analysis Failed');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>Education Analysis Tool</h1>
      <p>
        This tool runs the Python analysis model on student data.
      </p>

      <button
        onClick={handleRunAnalysis}
        disabled={isRunning}
        style={{
            padding: '10px 20px',
            fontSize: '16px',
            backgroundColor: isRunning ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: isRunning ? 'not-allowed' : 'pointer'
        }}
      >
        {isRunning ? 'Running...' : 'Run Analysis'}
      </button>

      <div style={{ marginTop: '20px' }}>
        <h3>Status: {status}</h3>
      </div>

      <div style={{
          marginTop: '20px',
          backgroundColor: '#f5f5f5',
          padding: '10px',
          borderRadius: '5px',
          height: '400px',
          overflowY: 'scroll',
          fontFamily: 'monospace'
      }}>
        <h3>Logs:</h3>
        {logs.map((log, index) => (
          <div key={index} style={{ whiteSpace: 'pre-wrap' }}>{log}</div>
        ))}
      </div>
    </div>
  );
};

export default App;
