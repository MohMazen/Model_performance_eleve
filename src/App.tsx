import { useState } from 'react'

declare global {
  interface Window {
    electron: {
      selectFile: () => Promise<string | null>;
      analyzeData: (filePath: string) => Promise<any>;
      onAnalysisComplete: (callback: (data: any) => void) => void;
      onAnalysisError: (callback: (error: string) => void) => void;
    }
  }
}

function App() {
  const [filePath, setFilePath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSelectFile = async () => {
    try {
      const path = await window.electron.selectFile()
      if (path) {
        setFilePath(path)
        setResult(null)
        setError(null)
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleAnalyze = async () => {
    if (!filePath) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      await window.electron.analyzeData(filePath)
    } catch (err: any) {
      setError(err.message)
      setLoading(false)
    }
  }

  // Setup listeners
  useState(() => {
    window.electron.onAnalysisComplete((data) => {
      setResult(data)
      setLoading(false)
    })

    window.electron.onAnalysisError((err) => {
      setError(err)
      setLoading(false)
    })
  })

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
      <h1>Education Analysis</h1>

      <div style={{ marginBottom: '2rem' }}>
        <button onClick={handleSelectFile}>Select Excel File</button>
        {filePath && <p>Selected: {filePath}</p>}
      </div>

      <div style={{ marginBottom: '2rem' }}>
        <button onClick={handleAnalyze} disabled={!filePath || loading}>
          {loading ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      {error && (
        <div style={{ color: 'red', padding: '1rem', border: '1px solid red', borderRadius: '4px' }}>
          <h3>Error:</h3>
          <pre>{error}</pre>
        </div>
      )}

      {result && (
        <div style={{ padding: '1rem', border: '1px solid #ccc', borderRadius: '4px', whiteSpace: 'pre-wrap' }}>
          <h3>Results:</h3>
          <div>{result}</div>
        </div>
      )}
    </div>
  )
}

export default App
