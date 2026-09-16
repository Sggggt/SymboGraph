@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-app.ps1" %*
set "START_APP_EXIT_CODE=%ERRORLEVEL%"
echo.
if "%START_APP_EXIT_CODE%"=="0" (
  echo [READY] SymboGraph startup completed.
  echo The backend and Web processes will keep running after this window closes.
) else (
  echo [FAILED] SymboGraph startup exited with code %START_APP_EXIT_CODE%.
  echo Review the error shown above before closing this window.
)
if /I not "%SYMBOGRAPH_NO_PAUSE%"=="1" (
  echo.
  echo Press any key to close this launcher window...
  pause >nul
)
exit /b %START_APP_EXIT_CODE%
