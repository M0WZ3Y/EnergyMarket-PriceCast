#!/bin/bash
# Quick Setup Script for Electricity Price Forecasting Thesis Project (Linux/Mac)

echo "========================================"
echo "Electricity Price Forecasting Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.11 or higher from https://python.org"
    exit 1
fi

echo "Python found. Starting setup..."
echo

# Run the setup script
python3 06_deployment/scripts/setup_environment.py

if [ $? -ne 0 ]; then
    echo
    echo "Setup failed. Please check the error messages above."
    exit 1
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Start Jupyter Notebook:"
echo "   jupyter notebook"
echo
echo "3. Open the data exploration notebook:"
echo "   06_deployment/notebooks/data_exploration/01_data_overview.ipynb"
echo
echo "Happy coding! 🚀"
echo