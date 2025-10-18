@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv not found. Run setup.bat first.
  pause & exit /b 1
)

call "venv\Scripts\activate.bat"
python -m core.utils.manage_indexes
endlocal
