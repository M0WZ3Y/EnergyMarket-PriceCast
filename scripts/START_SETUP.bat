@echo off
REM Simple Setup Launcher for Electricity Price Forecasting Thesis Project

echo ========================================
echo Electricity Price Forecasting Setup
echo ========================================
echo.

REM Try the quick setup first
if exist quick_setup.py (
    echo Running quick setup...
    python quick_setup.py
    if errorlevel 1 (
        echo.
        echo Quick setup failed, trying manual setup...
        goto manual_setup
    ) else (
        goto success
    )
)

:manual_setup
REM Manual setup as fallback
echo.
echo Running manual setup...
if exist 06_deployment\scripts\setup_environment.py (
    python 06_deployment\scripts\setup_environment.py
) else (
    echo ERROR: Setup scripts not found!
    echo Please make sure you're in the project directory.
    pause
    exit /b 1
)

if errorlevel 1 (
    goto error
) else (
    goto success
)

:success
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
echo    06_deployment\notebooks\data_exploration\01_data_overview.ipynb
echo.
echo Happy coding! 🚀
echo.
pause
exit /b 0

:error
echo.
echo ========================================
echo Setup failed!
echo ========================================
echo.
echo Please check the error messages above.
echo You can also try manual setup:
echo.
echo 1. Create virtual environment:
echo    python -m venv venv
echo.
echo 2. Activate it:
echo    venv\Scripts\activate
echo.
echo 3. Install dependencies:
echo    pip install -r 06_deployment\requirements\requirements.txt
echo.
pause
exit /b 1