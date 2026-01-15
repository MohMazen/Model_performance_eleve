"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('electron', {
    runAnalysis: (args) => electron_1.ipcRenderer.invoke('run-analysis', args),
    onLog: (callback) => electron_1.ipcRenderer.on('analysis-log', (_event, value) => callback(value)),
    onError: (callback) => electron_1.ipcRenderer.on('analysis-error', (_event, value) => callback(value)),
});
