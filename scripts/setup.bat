@echo off
REM Quick Setup Script for Electricity Price Forecasting Thesis Project (Windows)

REM Change to the script's directory
cd /d "%~dp0"

echo ========================================
echo Electricity Price Forecasting Setup
echo ========================================
echo Current directory: %CD%
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found. Starting setup...
echo.

REM Run the setup script
python "%~dp0\06_deployment\scripts\setup_environment.py"

if errorlevel 1 (
    echo.
    echo Setup failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Activate virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Start Jupyter Notebook:
echo    jupyter notebook
echo.
echo 3. Open the data exploration notebook:
echo    06_deployment/notebooks/data_exploration/01_data_overview.ipynb
echo.
echo Happy coding! 🚀
echo.
pause