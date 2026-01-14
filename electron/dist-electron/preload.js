"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('electron', {
    selectFile: () => electron_1.ipcRenderer.invoke('select-file'),
    runAnalysis: (filePath) => electron_1.ipcRenderer.invoke('run-analysis', filePath),
    onAnalysisLog: (callback) => {
        electron_1.ipcRenderer.on('analysis-log', (_event, message) => callback(message));
    },
    onAnalysisError: (callback) => {
        electron_1.ipcRenderer.on('analysis-error', (_event, message) => callback(message));
    },
    // Remove listeners to avoid memory leaks if needed
    removeAnalysisListeners: () => {
        electron_1.ipcRenderer.removeAllListeners('analysis-log');
        electron_1.ipcRenderer.removeAllListeners('analysis-error');
    }
});
//# sourceMappingURL=preload.js.map