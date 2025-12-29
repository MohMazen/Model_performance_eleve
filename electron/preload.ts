import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electron', {
  selectFile: () => ipcRenderer.invoke('select-file'),
  analyzeData: (filePath: string) => ipcRenderer.invoke('analyze-data', filePath),
  onAnalysisComplete: (callback: (data: any) => void) =>
    ipcRenderer.on('analysis-complete', (_, data) => callback(data)),
  onAnalysisError: (callback: (error: string) => void) =>
    ipcRenderer.on('analysis-error', (_, error) => callback(error)),
});
