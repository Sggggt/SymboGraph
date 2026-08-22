@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-app.ps1" %*
set "START_APP_EXIT_CODE=%ERRORLEVEL%"
if not "%START_APP_EXIT_CODE%"=="0" pause
exit /b %START_APP_EXIT_CODE%
