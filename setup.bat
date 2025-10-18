@echo off
setlocal EnableExtensions EnableDelayedExpansion
title WindowsContextualSearch - Setup

REM go to repo root (folder of this script)
cd /d "%~dp0"

echo ===============================================================
echo  WindowsContextualSearch - First-time Setup
echo ===============================================================
echo.

REM ---- Check Python ------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  echo Install Python 3.10+ first: https://www.python.org/downloads/
  echo.
  pause
  exit /b 1
)

REM ---- Create venv -------------------------------------------------
set "VENV_DIR=venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [SETUP] Creating virtual environment: %VENV_DIR%
  python -m venv "%VENV_DIR%"
) else (
  echo [OK] Virtual environment already exists.
)

echo [SETUP] Upgrading pip and installing requirements...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if exist "requirements.txt" (
  "%VENV_DIR%\Scripts\python.exe" -m pip install -r "requirements.txt"
) else (
  echo [WARN] requirements.txt not found. If this is intentional, ignore.
)

REM ---- Check Ollama -----------------------------------------------
where ollama >nul 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] Ollama is not installed or not in PATH.
  echo         Install from: https://ollama.com/download
  echo.
  start "" "https://ollama.com/download"
  pause
  exit /b 1
)

REM ---- Ask models (with sensible defaults) ------------------------
set "OLLAMA_CHAT_MODEL=llama3"
set "OLLAMA_EMBED_MODEL=nomic-embed-text"
echo.
echo Chat model to pull [default: %OLLAMA_CHAT_MODEL%] (press ENTER for default):
set /p "OLLAMA_CHAT_MODEL_IN=>"
if not "%OLLAMA_CHAT_MODEL_IN%"=="" set "OLLAMA_CHAT_MODEL=%OLLAMA_CHAT_MODEL_IN%"

echo Embedding model to pull [default: %OLLAMA_EMBED_MODEL%] (press ENTER for default):
set /p "OLLAMA_EMBED_MODEL_IN=>"
if not "%OLLAMA_EMBED_MODEL_IN%"=="" set "OLLAMA_EMBED_MODEL=%OLLAMA_EMBED_MODEL_IN%"

echo.
echo [SETUP] Pulling Ollama models (this may take a few minutes)...
ollama pull "%OLLAMA_CHAT_MODEL%" || echo [WARN] Could not pull %OLLAMA_CHAT_MODEL% (is ollama serve running?)
ollama pull "%OLLAMA_EMBED_MODEL%" || echo [WARN] Could not pull %OLLAMA_EMBED_MODEL% (is ollama serve running?)

REM ---- Pick data folder (File Explorer picker) --------------------
echo.
echo [SETUP] Select your data folder (PDF/TXT/MD). A dialog will open...
for /f "usebackq delims=" %%I in (`
  powershell -NoProfile -Command ^
    "$app = New-Object -ComObject Shell.Application; $folder = $app.BrowseForFolder(0, 'Select your data folder (PDF/TXT/MD)', 0, 0); if ($folder) { $path = $folder.Self.Path; [Console]::WriteLine($path) }"
`) do set "DATA_ROOT=%%I"

if not defined DATA_ROOT (
  echo [INFO] No folder picked. Using default .\data
  set "DATA_ROOT=%cd%\data"
)

echo [OK] Data folder: "%DATA_ROOT%"

REM ---- Persist workspace for run.bat and manage_indexes.py ---------
> "wcs_workspace.txt" echo "%DATA_ROOT%"
if not exist "index_store" mkdir "index_store"
> "index_store\workspaces.json" (
  echo [
  echo   "%DATA_ROOT%"
  echo ]
)

REM ---- Build indexes ----------------------------------------------
echo.
echo [INDEX] Building BM25 index...
"%VENV_DIR%\Scripts\python.exe" core/indexing/index_bm25.py --root "%DATA_ROOT%" || echo [WARN] BM25 index failed.

echo.
echo [INDEX] Building Chroma (vector) index...
"%VENV_DIR%\Scripts\python.exe" core/indexing/index_chroma.py --root "%DATA_ROOT%" || echo [WARN] Chroma index failed.

echo.
echo ===============================================================
echo  Setup complete.
echo  Next: double-click run.bat to start the API and open the UI.
echo ===============================================================
echo.
pause
endlocal
exit /b 0