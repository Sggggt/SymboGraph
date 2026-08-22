@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0rebuild-images.ps1" %*
set "REBUILD_IMAGES_EXIT_CODE=%ERRORLEVEL%"
if not "%REBUILD_IMAGES_EXIT_CODE%"=="0" pause
exit /b %REBUILD_IMAGES_EXIT_CODE%
