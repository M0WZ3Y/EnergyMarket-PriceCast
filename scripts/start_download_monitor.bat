@echo off
title PJM Enhanced Download Monitor - Zero Traffic Detection
echo Starting Enhanced PJM Download Monitor with Zero-Traffic Detection...
echo.
echo Features:
echo - Real-time download progress monitoring
echo - Zero-traffic detection (alerts after 3 minutes of no activity)
echo - Color-coded status indicators
echo - Connection issue warnings
echo - Updates every 30 seconds
echo.
echo You can minimize this window and the monitoring will continue.
echo Press Ctrl+C in this window to stop monitoring.
echo.
cd /d "%~dp0"
python enhanced_pjm_download_monitor.py
pause