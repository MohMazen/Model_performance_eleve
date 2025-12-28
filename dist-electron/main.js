"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
const child_process_1 = require("child_process");
const fs = __importStar(require("fs"));
const auto_launch_1 = __importDefault(require("auto-launch"));
// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    electron_1.app.quit();
}
let mainWindow = null;
let pythonProcess = null;
function createWindow() {
    mainWindow = new electron_1.BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
        },
    });
    const isDev = process.env.NODE_ENV === 'development';
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    }
    else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}
function startPythonProcess() {
    // Path to python script
    // In dev, it's relative to dist-electron (which is one level inside root). So we go up one level to root.
    // Wait, dist-electron is at root. __dirname is inside dist-electron.
    // So path.join(__dirname, '../Claude4_model/...')
    let scriptPath = path.join(__dirname, '../Claude4_model/education_analysisC4.py');
    if (electron_1.app.isPackaged) {
        scriptPath = path.join(process.resourcesPath, 'Claude4_model', 'education_analysisC4.py');
    }
    // Check if python is available or bundle it. For this task, we assume python is in PATH or use a venv.
    // Using 'python' or 'python3' depending on system.
    const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';
    console.log(`Starting python process: ${pythonExecutable} ${scriptPath}`);
    if (fs.existsSync(scriptPath)) {
        pythonProcess = (0, child_process_1.spawn)(pythonExecutable, [scriptPath]);
        pythonProcess.stdout?.on('data', (data) => {
            console.log(`Python stdout: ${data}`);
        });
        pythonProcess.stderr?.on('data', (data) => {
            console.error(`Python stderr: ${data}`);
        });
        pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
        });
    }
    else {
        console.error(`Python script not found at ${scriptPath}`);
    }
}
electron_1.app.on('ready', () => {
    createWindow();
    startPythonProcess();
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
electron_1.app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
electron_1.app.on('will-quit', () => {
    if (pythonProcess) {
        pythonProcess.kill();
    }
});
// IPC listeners for auto-launch
electron_1.ipcMain.handle('get-app-path', () => {
    return electron_1.app.getPath('exe');
});
// Configure auto-launch
const autoLaunch = new auto_launch_1.default({
    name: 'Education Analysis App',
    path: electron_1.app.getPath('exe'),
});
electron_1.ipcMain.handle('enable-auto-launch', async () => {
    try {
        await autoLaunch.enable();
        return true;
    }
    catch (error) {
        console.error('Failed to enable auto-launch:', error);
        return false;
    }
});
electron_1.ipcMain.handle('disable-auto-launch', async () => {
    try {
        await autoLaunch.disable();
        return true;
    }
    catch (error) {
        console.error('Failed to disable auto-launch:', error);
        return false;
    }
});
electron_1.ipcMain.handle('is-auto-launch-enabled', async () => {
    try {
        return await autoLaunch.isEnabled();
    }
    catch (error) {
        console.error('Failed to check auto-launch status:', error);
        return false;
    }
});
