import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  selectFile: () => ipcRenderer.invoke('select-file'),
  runAnalysis: (filePath: string) => ipcRenderer.invoke('run-analysis', filePath),
  onAnalysisLog: (callback: (message: string) => void) => ipcRenderer.on('analysis-log', (_event, value) => callback(value)),
  onAnalysisError: (callback: (message: string) => void) => ipcRenderer.on('analysis-error', (_event, value) => callback(value)),
});
