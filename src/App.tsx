import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

function App() {
  const [file, setFile] = useState<string | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [result, setResult] = useState<{ report: string; images: string[] } | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Setup listeners
    window.electron.onAnalysisLog((msg) => {
      setLogs((prev) => [...prev, msg])
    })
    window.electron.onAnalysisError((msg) => {
      setLogs((prev) => [...prev, `ERROR: ${msg}`])
    })

    return () => {
        window.electron.removeAnalysisListeners()
    }
  }, [])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleSelectFile = async () => {
    const filePath = await window.electron.selectFile()
    if (filePath) {
      setFile(filePath)
      setResult(null)
      setLogs([])
    }
  }

  const handleRunAnalysis = async () => {
    if (!file && !confirm("No file selected. Run with synthetic data?")) return;

    setAnalyzing(true)
    setLogs([])
    setResult(null)

    try {
      const data = await window.electron.runAnalysis(file)
      if (data.success) {
        setResult({ report: data.report, images: data.images })
      } else {
        alert('Analysis failed. Check logs.')
      }
    } catch (error) {
      console.error(error)
      alert(`Error: ${error}`)
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div className="container">
      <h1>Education Data Analysis</h1>

      <div className="controls">
        <button onClick={handleSelectFile} disabled={analyzing}>
          {file ? 'Change File' : 'Select Excel File'}
        </button>
        <span className="file-path">{file || 'No file selected (using synthetic data)'}</span>

        <button className="primary" onClick={handleRunAnalysis} disabled={analyzing}>
          {analyzing ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      <div className="main-content">
        <div className="logs-panel">
            <h3>Logs</h3>
            <div className="logs-container">
                {logs.map((log, i) => (
                    <div key={i} className="log-entry">{log}</div>
                ))}
                <div ref={logsEndRef} />
            </div>
        </div>

        {result && (
            <div className="results-panel">
                <h2>Analysis Report</h2>
                <div className="report-content">
                    <ReactMarkdown>{result.report}</ReactMarkdown>
                </div>

                <h3>Visualizations</h3>
                <div className="images-grid">
                    {result.images.map((img, i) => (
                        <div key={i} className="image-card">
                            <img src={`file://${img}`} alt={`Visualization ${i}`} />
                        </div>
                    ))}
                </div>
            </div>
        )}
      </div>
    </div>
  )
}

export default App
