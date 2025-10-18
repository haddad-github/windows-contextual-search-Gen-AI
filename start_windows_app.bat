@echo off
setlocal EnableExtensions EnableDelayedExpansion
title WindowsContextualSearch - Launcher

REM ----- Resolve repo root (folder of this script)
cd /d "%~dp0"

REM ----- Config (feel free to tweak)
set API_HOST=127.0.0.1
set API_PORT=8000
set VENV_DIR=venv

echo ===============================================================
echo  WindowsContextualSearch - Starting backend + app
echo ===============================================================

REM ----- Sanity checks
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [ERROR] Python venv not found at %VENV_DIR%
  echo Run setup.bat once on this machine to create it
  pause
  exit /b 1
)

REM ----- Try to start Ollama if needed
ollama list >nul 2>nul
if errorlevel 1 (
  echo [INFO] Starting Ollama service window...
  start "Ollama" cmd /k ollama serve
) else (
  echo [OK] Ollama is reachable
)

REM ----- Ensure a workspace is set (ask once, then persist)
set WORKSPACE_FILE=wcs_workspace.txt
set DATA_ROOT=
if exist "%WORKSPACE_FILE%" set /p DATA_ROOT=<"%WORKSPACE_FILE%"
if not defined DATA_ROOT (
  echo [PICK] Choose your data folder (PDF/TXT/MD)
  for /f "usebackq delims=" %%I in (`
    powershell -NoProfile -Command ^
      "$f=New-Object -ComObject Shell.Application;" ^
      "$p=$f.BrowseForFolder(0,'Select data folder',0,0);" ^
      "if($p){[Console]::WriteLine($p.Self.Path)}"
  `) do set DATA_ROOT=%%I

  if not defined DATA_ROOT (
    echo [INFO] No folder selected. Using .\data
    set DATA_ROOT=%cd%\data
  )
  > "%WORKSPACE_FILE%" echo %DATA_ROOT%
)

echo API: http://%API_HOST%:%API_PORT%
echo Workspace: %DATA_ROOT%
echo.

REM ----- Start the FastAPI server in a new window
echo [RUN] API server starting...
set ACTIVATE=call "%VENV_DIR%\Scripts\activate.bat"
set UVICORN=uvicorn api.server:app --host %API_HOST% --port %API_PORT%
start "WCS API" cmd /k %ACTIVATE% ^& %UVICORN%

REM small wait so the API is ready before the app opens
timeout /t 2 >nul

REM ----- Launch the WPF app
if exist "gui\WindowsContextualSearchApp\WindowsContextualSearchApp\bin\Release\net8.0-windows\WindowsContextualSearchApp.exe" (
  start "" "gui\WindowsContextualSearchApp\WindowsContextualSearchApp\bin\Release\net8.0-windows\WindowsContextualSearchApp.exe"
) else if exist "WindowsContextualSearchApp.exe" (
  start "" "WindowsContextualSearchApp.exe"
) else (
  echo [WARN] Could not find WindowsContextualSearchApp.exe
)

echo.
echo [OK] Launcher done. You can close this window if you want
echo.
endlocal
