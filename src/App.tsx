import React, { useState } from 'react';
import './App.css';

// Type definition for exposed electron API
declare global {
  interface Window {
    electronAPI: {
      getAppPath: () => Promise<string>;
      enableAutoLaunch: () => Promise<boolean>;
      disableAutoLaunch: () => Promise<boolean>;
      isAutoLaunchEnabled: () => Promise<boolean>;
    };
  }
}

function App() {
  const [path, setPath] = useState<string>('');
  const [autoLaunchEnabled, setAutoLaunchEnabled] = useState<boolean>(false);

  React.useEffect(() => {
    if (window.electronAPI) {
        window.electronAPI.isAutoLaunchEnabled().then(setAutoLaunchEnabled);
    }
  }, []);

  const handleGetPath = async () => {
    if (window.electronAPI) {
      const appPath = await window.electronAPI.getAppPath();
      setPath(appPath);
    }
  };

  const toggleAutoLaunch = async () => {
      if (window.electronAPI) {
          if (autoLaunchEnabled) {
              await window.electronAPI.disableAutoLaunch();
              setAutoLaunchEnabled(false);
          } else {
              await window.electronAPI.enableAutoLaunch();
              setAutoLaunchEnabled(true);
          }
      }
  };

  return (
    <div className="container">
      <h1>Education Analysis App</h1>
      <p>This is a React/Electron app spawning a Python backend.</p>
      <button onClick={handleGetPath}>Get App Path</button>
      {path && <p>App Path: {path}</p>}

      <div style={{margin: '20px 0'}}>
          <label>
              <input
                  type="checkbox"
                  checked={autoLaunchEnabled}
                  onChange={toggleAutoLaunch}
              />
              Launch on Startup
          </label>
      </div>

      <div style={{marginTop: '20px', border: '1px solid #ccc', padding: '10px'}}>
        <h2>Questionnaire Placeholder</h2>
        <p>The original questionnaire content would go here.</p>
        {/* We can port the Questionnaire.html content here later if requested */}
      </div>
    </div>
  );
}

export default App;
