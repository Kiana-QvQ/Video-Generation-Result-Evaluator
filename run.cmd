@echo off
setlocal
where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Run setup.ps1 first.
    exit /b 1
)
python "%~dp0start.py"
exit /b %ERRORLEVEL%
