@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Air Quality Dashboard

set "PY=.venv\Scripts\python.exe"

rem A venv whose base Python was moved/uninstalled is broken: rebuild it.
if exist "%PY%" (
    "%PY%" -c "import sys" >nul 2>nul || rmdir /s /q ".venv"
)

if not exist "%PY%" (
    echo Setting up environment for the first time...
    set "BASEPY="
    rem Pick the first Python that actually runs (the py launcher can point to a missing file).
    for %%C in ("py -3" "D:\Python\Python312\python.exe" "%LocalAppData%\Programs\Python\Python312\python.exe" "%LocalAppData%\Programs\Python\Python313\python.exe" "C:\Python312\python.exe" "python" "python3") do (
        if not defined BASEPY (
            %%~C -c "import venv, sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul && set "BASEPY=%%~C"
        )
    )
    if not defined BASEPY (
        echo Could not find a working Python 3.9+ installation.
        echo Install it from https://www.python.org/downloads/ and run this again.
        pause
        exit /b 1
    )
    echo Using !BASEPY!
    !BASEPY! -m venv .venv
    if not exist "%PY%" (
        echo Could not create virtual environment.
        pause
        exit /b 1
    )
)

"%PY%" -c "import streamlit, sklearn, plotly, pandas" >nul 2>nul
if errorlevel 1 (
    echo Installing requirements...
    "%PY%" -m pip install -q -r requirements.txt
)

set PORT=8501
echo Starting dashboard at http://localhost:%PORT% ...

rem Open Chrome once the server has had a moment to start.
start "" /b cmd /c "ping -n 6 127.0.0.1 >nul & (start chrome http://localhost:%PORT% || start http://localhost:%PORT%)"

"%PY%" -m streamlit run app.py --server.port %PORT% --server.headless true --browser.gatherUsageStats false

pause
