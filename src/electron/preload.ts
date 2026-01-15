import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electron', {
  runAnalysis: (args: any) => ipcRenderer.invoke('run-analysis', args),
  onLog: (callback: (log: string) => void) => ipcRenderer.on('analysis-log', (_event, value) => callback(value)),
  onError: (callback: (err: string) => void) => ipcRenderer.on('analysis-error', (_event, value) => callback(value)),
});
