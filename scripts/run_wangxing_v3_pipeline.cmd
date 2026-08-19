@echo off
setlocal
set "ROOT=%~dp0.."
"%ROOT%\.venv\Scripts\python.exe" "%~dp0run_wangxing_v3_pipeline.py" %*
exit /b %ERRORLEVEL%
