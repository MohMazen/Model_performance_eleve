import { app, BrowserWindow, ipcMain, shell } from 'electron';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

let mainWindow: BrowserWindow | null = null;

const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
};

app.on('ready', () => {
  createWindow();

  // Configure auto-launch
  const appFolder = path.dirname(process.execPath);
  const updateExe = path.resolve(appFolder, '..', 'Update.exe');
  const exeName = path.basename(process.execPath);

  app.setLoginItemSettings({
    openAtLogin: true,
    path: process.execPath,
    args: [
      '--process-start-args',
      `"--hidden"`
    ]
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// IPC Handler to run analysis
ipcMain.handle('run-analysis', async (event, args) => {
  return new Promise((resolve, reject) => {
    // Determine path to python script
    // In development, it's in Claude4_model/education_analysisC4.py
    // In production, we need to make sure it's bundled.
    let scriptPath = '';
    let pythonPath = 'python'; // Assume python is in path for now, or use a bundled one.

    // Determine a writable output directory
    const outputDir = app.getPath('userData');

    if (process.env.NODE_ENV === 'development') {
       scriptPath = path.join(__dirname, '../Claude4_model/education_analysisC4.py');
    } else {
       // In production, resources are usually in resources/app.asar.unpacked or resources/
       // We will configure electron-builder to put Claude4_model in extraResources
       scriptPath = path.join(process.resourcesPath, 'Claude4_model', 'education_analysisC4.py');
    }

    const pyProcess = spawn(pythonPath, [scriptPath, '--output-dir', outputDir], {
        cwd: path.dirname(scriptPath)
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
      } else {
        reject({ success: false, error, code });
      }
    });
  });
});
