@echo off
setlocal

set "PROJECT_ROOT=%~dp0..\.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo Running Wang Xing specialization training from:
echo   %PROJECT_ROOT%
echo Long-running identity extraction stays in the foreground.

set "SEEDANCE_MANIFEST=%PROJECT_ROOT%\data\au\WangXing_Seedance\pseudo_expression_manifest.json"
if exist "%SEEDANCE_MANIFEST%" (
  "%PYTHON%" "%PROJECT_ROOT%\scripts\wangxing\train_wangxing_specialization.py" %* --seedance-label-manifest "%SEEDANCE_MANIFEST%"
) else (
  echo Warning: Seedance pseudo-label manifest is missing.
  echo Run scripts\wangxing\run_seedance_expression_labeling.cmd first.
  "%PYTHON%" "%PROJECT_ROOT%\scripts\wangxing\train_wangxing_specialization.py" %*
)
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
  echo Training failed with exit code %EXIT_CODE%.
)
exit /b %EXIT_CODE%
