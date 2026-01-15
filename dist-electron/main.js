"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path_1 = __importDefault(require("path"));
const child_process_1 = require("child_process");
// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
    electron_1.app.quit();
}
let mainWindow = null;
const createWindow = () => {
    mainWindow = new electron_1.BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path_1.default.join(__dirname, 'preload.js'),
            nodeIntegration: false,
            contextIsolation: true,
        },
    });
    if (process.env.NODE_ENV === 'development') {
        mainWindow.loadURL('http://localhost:5173');
        mainWindow.webContents.openDevTools();
    }
    else {
        mainWindow.loadFile(path_1.default.join(__dirname, '../dist/index.html'));
    }
};
electron_1.app.on('ready', () => {
    createWindow();
    // Configure auto-launch
    const appFolder = path_1.default.dirname(process.execPath);
    const updateExe = path_1.default.resolve(appFolder, '..', 'Update.exe');
    const exeName = path_1.default.basename(process.execPath);
    electron_1.app.setLoginItemSettings({
        openAtLogin: true,
        path: process.execPath,
        args: [
            '--process-start-args',
            `"--hidden"`
        ]
    });
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
electron_1.app.on('activate', () => {
    if (electron_1.BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});
// IPC Handler to run analysis
electron_1.ipcMain.handle('run-analysis', async (event, args) => {
    return new Promise((resolve, reject) => {
        // Determine path to python script
        // In development, it's in Claude4_model/education_analysisC4.py
        // In production, we need to make sure it's bundled.
        let scriptPath = '';
        let pythonPath = 'python'; // Assume python is in path for now, or use a bundled one.
        // Determine a writable output directory
        const outputDir = electron_1.app.getPath('userData');
        if (process.env.NODE_ENV === 'development') {
            scriptPath = path_1.default.join(__dirname, '../Claude4_model/education_analysisC4.py');
        }
        else {
            // In production, resources are usually in resources/app.asar.unpacked or resources/
            // We will configure electron-builder to put Claude4_model in extraResources
            scriptPath = path_1.default.join(process.resourcesPath, 'Claude4_model', 'education_analysisC4.py');
        }
        const pyProcess = (0, child_process_1.spawn)(pythonPath, [scriptPath, '--output-dir', outputDir], {
            cwd: path_1.default.dirname(scriptPath)
        });
        let output = '';
        let error = '';
        pyProcess.stdout.on('data', (data) => {
            output += data.toString();
            // Send progress updates if possible
            if (mainWindow) {
                mainWindow.webContents.send('analysis-log', data.toString());
            }
        });
        pyProcess.stderr.on('data', (data) => {
            error += data.toString();
            if (mainWindow) {
                mainWindow.webContents.send('analysis-error', data.toString());
            }
        });
        pyProcess.on('close', (code) => {
            if (code === 0) {
                resolve({ success: true, output });
            }
            else {
                reject({ success: false, error, code });
            }
        });
    });
});
