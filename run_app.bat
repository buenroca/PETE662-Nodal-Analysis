@echo off
REM PETE 662 Nodal Analysis - double-click to start the app in your browser.
cd /d "%~dp0"
python -m streamlit run app.py
pause
