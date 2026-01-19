# Electron Application Guide

This guide explains how to develop and build the Education Analysis Electron application.

## Development

To start the application in development mode:

```bash
npm install
npm run dev
```

This will start the Vite development server and launch the Electron application. The backend Python script (`Claude4_model/education_analysisC4.py`) is executed directly by the Electron main process.

**Note:** You must have Python installed and the required dependencies installed:

```bash
pip install -r Claude4_model/requirements_txt.txt
```

## Building for Production

To build a standalone Windows executable (installer):

```bash
npm run build
```

This command performs the following steps:
1. Compiles the React frontend.
2. Compiles the Electron main process.
3. Packages the application using `electron-builder`.

The output will be in the `release/` directory.

### Python Dependency

The application expects `python` to be available in the system PATH on the user's machine. The Python script and `Claude4_model` folder are bundled with the application (in `resources/Claude4_model`).

If you wish to make the application truly standalone (without requiring the user to install Python), you should compile the Python script into an executable using `pyinstaller` and update `electron/main.ts` to point to the executable.

To compile with PyInstaller (example):
```bash
pyinstaller --onefile --name education_analysis Claude4_model/education_analysisC4.py
```

Then update `package.json` `extraResources` to include the generated executable instead of the `.py` file, and update `electron/main.ts` logic to spawn the executable.

## Auto-Launch

The application is configured to launch automatically on Windows startup. This is handled in `electron/main.ts` using `app.setLoginItemSettings`.
