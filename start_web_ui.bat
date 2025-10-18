@echo off
setlocal EnableExtensions EnableDelayedExpansion
title WindowsContextualSearch - Run

cd /d "%~dp0"

set "VENV_DIR=venv"
set "DEFAULT_HOST=127.0.0.1"
set "DEFAULT_PORT=8000"

echo ===============================================================
echo  WindowsContextualSearch - Run Configuration
echo ===============================================================
echo.

set /p "API_HOST=Enter host [default: %DEFAULT_HOST%] (press ENTER for default): "
if "%API_HOST%"=="" set "API_HOST=%DEFAULT_HOST%"

set /p "API_PORT=Enter port [default: %DEFAULT_PORT%] (press ENTER for default): "
if "%API_PORT%"=="" set "API_PORT=%DEFAULT_PORT%"

echo.
echo Using:
echo   Host: %API_HOST%
echo   Port: %API_PORT%
echo ===============================================================
echo.

REM ---- Read stored workspace (optional) ---------------------------
set "DATA_ROOT="
if exist "wcs_workspace.txt" (
  for /f "usebackq delims=" %%A in (`type "wcs_workspace.txt"`) do set "DATA_ROOT=%%A"
)

echo API: http://%API_HOST%:%API_PORT%
if defined DATA_ROOT echo Workspace: "%DATA_ROOT%"
echo.

REM ---- Sanity: venv ------------------------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [ERROR] venv not found. Run setup.bat first.
  echo.
  pause
  exit /b 1
)

REM ---- Quick Ollama check -----------------------------------------
ollama list >nul 2>nul
if errorlevel 1 (
  echo [INFO] Starting Ollama in a new window...
  start "Ollama" cmd /k ollama serve
  echo [WAIT] Giving Ollama a few seconds to start...
  timeout /t 5 /nobreak >nul
) else (
  echo [OK] Ollama is reachable.
)

REM ---- Start API in a new window ----------------------------------
echo [RUN] Starting API server (uvicorn) in a new window...
start "WCS API" cmd /k "%VENV_DIR%\Scripts\activate.bat" ^& uvicorn api.server:app --host %API_HOST% --port %API_PORT%

REM ---- Open Web UI -------------------------------------------------
if exist "web_ui\index.html" (
  echo [OPEN] Launching web UI...
  start "" "web_ui\index.html"
) else (
  echo [WARN] web_ui\index.html not found.
)

REM ---- Optionally launch WPF EXE if built -------------------------
set "WPF_EXE=gui\WindowsContextualSearchApp\bin\Release\net8.0-windows\WindowsContextualSearchApp.exe"
if exist "%WPF_EXE%" (
  echo [OPEN] Launching desktop app...
  start "" "%WPF_EXE%"
) else (
  echo [INFO] WPF EXE not found (build later in Visual Studio if desired).
)

echo.
echo [OK] All components launched. You can close this window anytime.
echo.
pause
endlocal
exit /b 0