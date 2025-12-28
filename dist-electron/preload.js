"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('electronAPI', {
    getAppPath: () => electron_1.ipcRenderer.invoke('get-app-path'),
    enableAutoLaunch: () => electron_1.ipcRenderer.invoke('enable-auto-launch'),
    disableAutoLaunch: () => electron_1.ipcRenderer.invoke('disable-auto-launch'),
    isAutoLaunchEnabled: () => electron_1.ipcRenderer.invoke('is-auto-launch-enabled'),
});
