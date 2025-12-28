import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import { spawn, ChildProcess } from 'child_process';
import * as fs from 'fs';
import AutoLaunch from 'auto-launch';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
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
  } else {
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

  if (app.isPackaged) {
    scriptPath = path.join(process.resourcesPath, 'Claude4_model', 'education_analysisC4.py');
  }

  // Check if python is available or bundle it. For this task, we assume python is in PATH or use a venv.
  // Using 'python' or 'python3' depending on system.
  const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';

  console.log(`Starting python process: ${pythonExecutable} ${scriptPath}`);

  if (fs.existsSync(scriptPath)) {
      pythonProcess = spawn(pythonExecutable, [scriptPath]);

      pythonProcess.stdout?.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
      });

      pythonProcess.stderr?.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
      });
  } else {
      console.error(`Python script not found at ${scriptPath}`);
  }
}

app.on('ready', () => {
  createWindow();
  startPythonProcess();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on('will-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

// IPC listeners for auto-launch
ipcMain.handle('get-app-path', () => {
    return app.getPath('exe');
});

// Configure auto-launch
const autoLaunch = new AutoLaunch({
  name: 'Education Analysis App',
  path: app.getPath('exe'),
});

ipcMain.handle('enable-auto-launch', async () => {
    try {
        await autoLaunch.enable();
        return true;
    } catch (error) {
        console.error('Failed to enable auto-launch:', error);
        return false;
    }
});

ipcMain.handle('disable-auto-launch', async () => {
    try {
        await autoLaunch.disable();
        return true;
    } catch (error) {
        console.error('Failed to disable auto-launch:', error);
        return false;
    }
});

ipcMain.handle('is-auto-launch-enabled', async () => {
    try {
        return await autoLaunch.isEnabled();
    } catch (error) {
        console.error('Failed to check auto-launch status:', error);
        return false;
    }
});
