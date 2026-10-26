@echo off
title Poisson Burst Dashboard
echo Starting Poisson Burst Dashboard...
cd /d "%~dp0poisson-burst-dashboard"
streamlit run app.py
pause
