import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import { spawn } from 'child_process';
import fs from 'fs';
import os from 'os';

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

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });

  // Auto-launch configuration
  const appPath = app.getPath('exe');
  // Simple auto-launch for Windows using LoginItemSettings
  // This works well for NSIS installers and generally for Windows
  app.setLoginItemSettings({
    openAtLogin: true,
    path: appPath,
    args: [] // Add any arguments if needed
  });

  console.log('Auto-launch configured for:', appPath);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('select-file', async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [{ name: 'Excel Files', extensions: ['xlsx', 'xls'] }],
  });
  return result.filePaths[0] || null;
});

ipcMain.handle('run-analysis', async (event, filePath) => {
  return new Promise((resolve, reject) => {
    // Determine the path to the python script or executable
    let pythonScriptPath;
    let pythonExecutable = 'python3'; // Default to python3 in path

    // In production, we might bundle the python script or executable
    if (app.isPackaged) {
        // Look in resources
        pythonScriptPath = path.join(process.resourcesPath, 'Claude4_model', 'education_analysisC4.py');
        // If we bundle a python executable, we'd point to it here.
        // For this guide, we assume python is installed on the system or we use the script.
        // A robust solution would use pyinstaller to create a standalone executable.
        // If using pyinstaller, pythonScriptPath would be the exe.

        // Let's assume we are shipping the .py file and relying on system python for now,
        // or the user has packaged it into an exe named 'education_analysisC4.exe' placed in resources.

        // Check if an exe exists (bundled via pyinstaller)
        const possibleExe = path.join(process.resourcesPath, 'education_analysisC4.exe');
        if (fs.existsSync(possibleExe)) {
            pythonExecutable = possibleExe;
            pythonScriptPath = ''; // No script argument needed if it's the exe
        } else {
             // Fallback to script
             // On Windows, might need 'python' instead of 'python3'
             pythonExecutable = 'python';
        }
    } else {
        // Development
        pythonScriptPath = path.join(__dirname, '../../Claude4_model/education_analysisC4.py');
        // Check if python3 exists, else try python
        // This is a bit loose, usually one knows the dev env.
    }

    const outputDir = path.join(app.getPath('userData'), 'analysis_output');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const args = [];
    if (pythonScriptPath) {
        args.push(pythonScriptPath);
    }

    if (filePath) {
        args.push('--file', filePath);
    }

    args.push('--output-dir', outputDir);

    console.log(`Spawning: ${pythonExecutable} ${args.join(' ')}`);

    const pythonProcess = spawn(pythonExecutable, args);

    let output = '';
    let errorOutput = '';

    pythonProcess.stdout.on('data', (data) => {
      const msg = data.toString();
      console.log(`Python stdout: ${msg}`);
      output += msg;
      // Send progress updates to renderer if needed
      if (mainWindow) {
          mainWindow.webContents.send('analysis-log', msg);
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const msg = data.toString();
      console.error(`Python stderr: ${msg}`);
      errorOutput += msg;
      if (mainWindow) {
          mainWindow.webContents.send('analysis-error', msg);
      }
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        // Read the generated markdown report
        const reportPath = path.join(outputDir, 'rapport_analyse_scolaire.md');
        let reportContent = '';
        try {
            reportContent = fs.readFileSync(reportPath, 'utf-8');
        } catch (e) {
            reportContent = 'Report generated but could not be read.';
        }

        // Get list of images
        const images = fs.readdirSync(outputDir)
            .filter(file => file.endsWith('.png'))
            .map(file => path.join(outputDir, file));

        resolve({ success: true, report: reportContent, images, outputDir });
      } else {
        reject({ success: false, error: errorOutput, code });
      }
    });

    pythonProcess.on('error', (err) => {
        reject({ success: false, error: err.message });
    });
  });
});
