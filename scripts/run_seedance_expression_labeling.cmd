@echo off
setlocal

set "PROJECT_ROOT=%~dp0.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo Labeling Seedance expression content from video features.
echo Long-running extraction stays in the foreground.

"%PYTHON%" "%PROJECT_ROOT%\scripts\label_seedance_expressions.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo Seedance labeling failed with exit code %EXIT_CODE%.
)
exit /b %EXIT_CODE%
