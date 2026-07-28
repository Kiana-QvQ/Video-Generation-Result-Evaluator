@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-grpc.ps1"
exit /b %ERRORLEVEL%
