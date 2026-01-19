import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

function createWindow() {
  const mainWindow = new BrowserWindow({
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
}

app.whenReady().then(() => {
  createWindow();

  // Configure auto-launch on startup
  // For NSIS installers, we just point to the executable itself
  if (process.platform === 'win32') {
      app.setLoginItemSettings({
        openAtLogin: true,
        path: process.execPath,
        args: ['--hidden'] // Optional: start hidden
      });
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Excel Files', extensions: ['xlsx', 'xls'] }],
  });
  return result.filePaths[0];
});

ipcMain.handle('run-analysis', async (event, filePath) => {
  return new Promise((resolve, reject) => {
    let pythonPath = 'python'; // Or specific path if bundled
    let scriptPath = '';

    if (app.isPackaged) {
      // In production, resources are in resources/Claude4_model
      scriptPath = path.join(process.resourcesPath, 'Claude4_model', 'education_analysisC4.py');
    } else {
      // In development, relative to project root
      scriptPath = path.join(__dirname, '../../Claude4_model/education_analysisC4.py');
    }

    console.log(`Running script: ${scriptPath} with file: ${filePath}`);

    // Verify script exists
    if (!fs.existsSync(scriptPath)) {
        reject(`Script not found at: ${scriptPath}`);
        return;
    }

    // Pass the file path as an argument to the python script
    // We also pass an output directory for generated files
    const outputDir = app.getPath('userData');
    const args = [scriptPath, '--file', filePath, '--output-dir', outputDir];

    const pythonProcess = spawn(pythonPath, args);

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      const str = data.toString();
      console.log('Python stdout:', str);
      output += str;
      // Send progress or logs to renderer if needed
      event.sender.send('analysis-log', str);
    });

    pythonProcess.stderr.on('data', (data) => {
      const str = data.toString();
      console.error('Python stderr:', str);
      errorOutput += str;
      event.sender.send('analysis-error', str);
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve({ success: true, output, outputDir });
      } else {
        reject(`Python process exited with code ${code}. Error: ${errorOutput}`);
      }
    });
  });
});
