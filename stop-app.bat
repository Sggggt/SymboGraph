@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-app.ps1" %*
set "STOP_APP_EXIT_CODE=%ERRORLEVEL%"
if not "%STOP_APP_EXIT_CODE%"=="0" pause
exit /b %STOP_APP_EXIT_CODE%
