@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "SIM_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%SIM_PYTHON%" set "SIM_PYTHON=%CD%\..\.venv\Scripts\python.exe"
if not exist "%SIM_PYTHON%" (
    echo Python wurde nicht gefunden: "%SIM_PYTHON%"
    echo Erstelle die Umgebung mit: python -m venv .venv
    popd
    exit /b 1
)
"%SIM_PYTHON%" "%CD%\main.py" %*
exit /b %errorlevel%
