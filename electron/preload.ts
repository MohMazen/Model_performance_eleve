import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  getAppPath: () => ipcRenderer.invoke('get-app-path'),
  enableAutoLaunch: () => ipcRenderer.invoke('enable-auto-launch'),
  disableAutoLaunch: () => ipcRenderer.invoke('disable-auto-launch'),
  isAutoLaunchEnabled: () => ipcRenderer.invoke('is-auto-launch-enabled'),
});
