@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-web.ps1" %*
set "START_WEB_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%START_WEB_EXIT_CODE%"=="0" (
  echo [READY] SymboGraph Web startup completed.
  echo The Web process will keep running after this window closes.
) else (
  echo [FAILED] SymboGraph Web startup exited with code %START_WEB_EXIT_CODE%.
  echo Review the error shown above before closing this window.
)
if /I not "%SYMBOGRAPH_NO_PAUSE%"=="1" (
  echo.
  echo Press any key to close this launcher window...
  pause >nul
)
exit /b %START_WEB_EXIT_CODE%
