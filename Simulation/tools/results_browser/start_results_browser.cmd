@echo off
setlocal
pushd "%~dp0..\.." || exit /b 1
set "SIM_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%SIM_PYTHON%" set "SIM_PYTHON=%CD%\..\.venv\Scripts\python.exe"
if not exist "%SIM_PYTHON%" (
    echo Python wurde nicht gefunden: "%SIM_PYTHON%"
    pause
    popd
    exit /b 1
)
"%SIM_PYTHON%" "%CD%\tools\results_browser\browse_results.py" --results "%CD%\results" %*
if errorlevel 1 pause
popd
endlocal
