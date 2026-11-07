@echo off
cd /d "%~dp0"
echo Installing required libraries (first time can take a few minutes)...
py -m pip install -r requirements.txt || python -m pip install -r requirements.txt
echo Starting QuantLab... your browser will open.
py -m streamlit run app.py || python -m streamlit run app.py
pause
