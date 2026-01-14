import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electron', {
  selectFile: () => ipcRenderer.invoke('select-file'),
  runAnalysis: (filePath: string) => ipcRenderer.invoke('run-analysis', filePath),
  onAnalysisLog: (callback: (message: string) => void) => {
    ipcRenderer.on('analysis-log', (_event, message) => callback(message));
  },
  onAnalysisError: (callback: (message: string) => void) => {
    ipcRenderer.on('analysis-error', (_event, message) => callback(message));
  },
  // Remove listeners to avoid memory leaks if needed
  removeAnalysisListeners: () => {
      ipcRenderer.removeAllListeners('analysis-log');
      ipcRenderer.removeAllListeners('analysis-error');
  }
});
