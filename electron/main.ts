import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import { spawn } from 'child_process';

// Auto-launch configuration using native API
const exePath = app.getPath('exe');
app.setLoginItemSettings({
  openAtLogin: true,
  path: exePath,
  args: [
    '--process-start-args', `"--hidden"`
  ]
});

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
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

app.whenReady().then(createWindow);

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

// IPC handlers
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow!, {
    properties: ['openFile'],
    filters: [{ name: 'Excel Files', extensions: ['xlsx', 'xls'] }],
  });
  return result.filePaths[0] || null;
});

ipcMain.handle('analyze-data', async (event, filePath) => {
  // Determine python path and script path
  // In production, we might expect a bundled python or use system python
  // We assume 'python' is available in PATH or packaged

  // For this implementation, we assume the script is in 'Claude4_model/education_analysisC4.py'
  // When packaged with electron-builder, we included 'Claude4_model' in extraResources

  const isDev = process.env.NODE_ENV === 'development';
  const scriptPath = isDev
    ? path.join(__dirname, '../Claude4_model/education_analysisC4.py')
    : path.join(process.resourcesPath, 'Claude4_model/education_analysisC4.py');

  const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';

  console.log(`Running python script: ${scriptPath} with file: ${filePath}`);

  const pythonProcess = spawn(pythonCommand, [scriptPath, '--file', filePath]);

  let output = '';
  let errorOutput = '';

  pythonProcess.stdout.on('data', (data) => {
    output += data.toString();
    console.log(`Python stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    errorOutput += data.toString();
    console.error(`Python stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code === 0) {
        // The script generates 'rapport_analyse_scolaire.md'. We could read it and send it back.
        // Or send the stdout.
        // Let's try to read the generated report if it exists.
        const reportPath = 'rapport_analyse_scolaire.md'; // The script writes to CWD by default

        // Wait, the script writes to current working directory.
        // We should probably pass the output directory or change cwd.
        // For now, let's send back the stdout which contains logs and maybe info.

        mainWindow?.webContents.send('analysis-complete', output + "\n\nAnalysis finished successfully.");
    } else {
      mainWindow?.webContents.send('analysis-error', `Python process exited with code ${code}\n${errorOutput}`);
    }
  });
});
